import numpy as np
import pandas as pd
import os.path
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from subprocess import check_output
from datetime import datetime
from catboost import CatBoostClassifier
from joblib import Parallel, delayed
import multiprocessing
from Faron.F1ExpMax import F1Optimizer

with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    import prince
    import xgboost as xgb
    from sklearn import linear_model
    from sklearn.metrics import r2_score
    from sklearn.svm import SVR, LinearSVR
    from sklearn.model_selection import train_test_split
    from sklearn.decomposition import PCA, FastICA, TruncatedSVD
    from sklearn.preprocessing import OneHotEncoder, LabelEncoder
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.naive_bayes import GaussianNB
    from sklearn.tree import DecisionTreeClassifier
color = sns.color_palette()

FEATURE_PATH = 'data/train_features.hdf'

models = {
    'LinReg': linear_model.LinearRegression(),
    'Lasso': linear_model.Lasso(alpha=0.005),
    'Ridge': linear_model.Ridge(alpha=0.005),
    'ElasticNet': linear_model.ElasticNet(alpha=0.005),
    'kNN': KNeighborsRegressor(n_neighbors=5),
    'RandForest': RandomForestRegressor(n_estimators=100, oob_score=True,
                                        max_features='auto'),
    'SVR': SVR(),
    'cat': CatBoostClassifier()
        }

xgb_params = {
    "objective": "reg:logistic",
    "eval_metric": "logloss",
    "eta": 0.1,
    "max_depth": 6,
    "min_child_weight": 10,
    "gamma": 0.70,
    "subsample": 0.76,
    "colsample_bytree": 0.95,
    "alpha": 2e-05,
    "lambda": 10
}

conversion_dict = {'product_id': np.uint16,
                   'aisle_id': np.uint8,
                   'department_id': np.uint8,
                   'order_id': np.uint32,
                   'add_to_cart_order': np.uint16,
                   'reordered': np.uint8,
                   'user_id': np.uint32,
                   'order_number': np.uint32,
                   'order_dow': np.uint8,
                   'order_hour_of_day': np.uint8,
                   'days_since_prior_order': np.float16,
                   }


def add_order_streak(df_):
    tmp = df_.copy()
    tmp.user_id = 1

    UP = tmp.pivot(index="product_id", columns='order_number').fillna(-1)
    UP.columns = UP.columns.droplevel(0)

    x = np.abs(UP.diff(axis=1).fillna(2)).values[:, ::-1]
    df_.set_index("product_id", inplace=True)
    df_['order_streak'] = np.multiply(np.argmax(x, axis=1) + 1, UP.iloc[:, -1])
    df_.reset_index(drop=False, inplace=True)
    return df_


def apply_parallel(df_groups, _func):
    nthreads = multiprocessing.cpu_count()  # >> 1
    print("nthreads: {}".format(nthreads))

    res = Parallel(n_jobs=nthreads)(delayed(_func)(grp.copy()) for _, grp in df_groups)
    return pd.concat(res)


def load_n_build_data():
    # aisles = pd.read_csv('data/aisles.csv', engine='c')
    # departments = pd.read_csv('data/departments.csv', engine='c')
    products = pd.read_csv('data/products.csv', engine='c',
                           dtype=conversion_dict)

    train_df = pd.read_csv('data/order_products__train.csv', engine='c',
                           dtype=conversion_dict)

    prior_df = pd.read_csv('data/order_products__prior.csv', engine='c',
                           dtype=conversion_dict)
    orders_df = pd.read_csv('data/orders.csv', engine='c', dtype=conversion_dict)
    # encode eval_set in orders_df
    lbl_enc = LabelEncoder()
    orders_df.loc[:, 'eval_set'] =\
        lbl_enc.fit_transform(orders_df.loc[:, 'eval_set']).astype(np.uint8)

    print('priors {}: {}'.format(prior_df.shape, ', '.join(prior_df.columns)))
    print('orders {}: {}'.format(orders_df.shape, ', '.join(orders_df.columns)))
    print('train {}: {}'.format(train_df.shape, ', '.join(train_df.columns)))

    print('add product features..')
    prods = pd.DataFrame()
    prods['orders'] = prior_df.groupby(prior_df.product_id).size()
    prods['reorders'] = prior_df['reordered'].groupby(prior_df.product_id).sum()
    prods['reorder_rate'] = (prods.reorders / prods.orders).astype(np.float32)
    products = products.join(prods, on='product_id')
    # there are products in products-dataframe that are not in prior or train
    products.replace([np.inf, -np.inf], np.nan, inplace=True)
    products.fillna(0, inplace=True)
    products.loc[:, 'orders'] = products.loc[:, 'orders'].astype(np.uint32)
    products.loc[:, 'reorders'] = products.loc[:, 'reorders'].astype(np.uint32)
    products.set_index('product_id', drop=False, inplace=True)
    del prods

    print('add order streak feature..')
    orders_ord_streak = orders_df.groupby(['user_id']).tail(5 + 1)
    prior_ord_streak = orders_ord_streak.merge(prior_df, how='inner',
                                               on="order_id")
    prior_ord_streak = prior_ord_streak[['user_id', 'product_id', 'order_number']]
    user_groups = prior_ord_streak.groupby('user_id')
    df_order_streak = apply_parallel(user_groups, add_order_streak)
    # df_ord_streak = pd.concat([add_order_streak(grp) for _, grp in user_groups])
    df_order_streak = \
        df_order_streak.drop("order_number", axis=1).drop_duplicates().\
            reset_index(drop=True)
    df_order_streak = df_order_streak.loc[:, ['user_id', 'product_id',
                                              'order_streak']]
    df_order_streak.loc[:, 'order_streak'] = \
        df_order_streak['order_streak'].astype(np.uint8)
    del orders_ord_streak
    del prior_ord_streak
    del user_groups

    print('add order info to prior..')
    orders_df.set_index('order_id', inplace=True, drop=False)
    priors = prior_df.join(orders_df, on='order_id', rsuffix='_')
    priors.drop('order_id_', inplace=True, axis=1)

    print('add user features..')
    usr = pd.DataFrame()
    usr['average_days_between_orders'] = \
        orders_df.groupby('user_id')['days_since_prior_order'].mean().astype(
            np.float32)
    usr['nb_orders'] = orders_df.groupby('user_id').size().astype(np.uint8)

    users = pd.DataFrame()
    users['total_items'] = priors.groupby('user_id').size().astype(np.uint16)
    users['all_products'] = priors.groupby('user_id')['product_id'].apply(set)
    users['total_distinct_items'] = \
        (users.all_products.map(len)).astype(np.uint16)

    users = users.join(usr)
    del usr
    users['average_basket'] = \
        (users.total_items / users.nb_orders).astype(np.float32)

    # userXproduct features (faster)
    print("add userxProduct features..")
    userXproduct = priors.copy()
    userXproduct['UP'] = \
        userXproduct.product_id.astype(np.uint64) + \
        userXproduct.user_id.astype(np.uint64) * 100000
    userXproduct = userXproduct.sort_values('order_number')
    userXproduct = userXproduct \
        .groupby('UP', sort=False) \
        .agg({'order_id': ['size', 'last'], 'add_to_cart_order': 'sum'})
    userXproduct.columns = ['nb_orders', 'last_order_id', 'sum_pos_in_cart']
    userXproduct.astype(
        {'nb_orders': np.uint16,
         'last_order_id': np.int8,
         'sum_pos_in_cart': np.uint32},
        inplace=True)
    del priors

    # train / test orders
    print('split orders : train, test')
    test_orders_ = orders_df[lbl_enc.inverse_transform(orders_df.eval_set) == 'test']
    train_orders_ = \
        orders_df[lbl_enc.inverse_transform(orders_df.eval_set) == 'train']

    train_df.set_index(['order_id', 'product_id'], inplace=True, drop=False)
    print(userXproduct.info())

    train_df = features(train_orders_, set(train_df.index),
                        users=users, orders_df=orders_df, products=products,
                        userXproduct=userXproduct,
                        df_order_streak=df_order_streak, labels_given=True)

    # build candidates list for test
    test_df = pd.read_csv('data/sample_submission.csv', engine='c',
                          dtype=conversion_dict)
    test_df = features(test_orders_, set(test_df.index),
                       users=users, orders_df=orders_df, products=products,
                       userXproduct=userXproduct,
                       df_order_streak=df_order_streak)

    print('save train features..')
    train_df.to_hdf(FEATURE_PATH, 'train')
    test_df.to_hdf(FEATURE_PATH, 'test')

    del users, orders_df, products, userXproduct, df_order_streak

    return train_df, test_df


# build list of candidate products to reorder, with features ###
def features(selected_orders, train_index, users, orders_df, products,
             userXproduct, df_order_streak, labels_given=False):
    print('build candidate list..')
    order_list = []
    product_list = []
    labels_ = []

    for row in selected_orders.itertuples():
        order_id = row.order_id
        user_id = row.user_id
        user_products = users.all_products[user_id]
        product_list += user_products
        order_list += [order_id] * len(user_products)
        if labels_given:
            labels_ += [(order_id, product) in train_index for product in
                        user_products]

    df = pd.DataFrame({'order_id': order_list, 'product_id': product_list},
                      dtype=np.uint32)
    if labels_given:
        df['y'] = np.array(labels_, dtype=np.uint8)

    del order_list
    del product_list

    print('user related features')
    df['user_id'] = df.order_id.map(orders_df.user_id)
    df['user_total_orders'] = df.user_id.map(users.nb_orders)
    df['user_total_items'] = df.user_id.map(users.total_items)
    df['total_distinct_items'] = df.user_id.map(users.total_distinct_items)
    df['user_average_days_between_orders'] = df.user_id.map(
        users.average_days_between_orders)
    df['user_average_basket'] = df.user_id.map(users.average_basket)

    print('order related features')
    df['dow'] = df.order_id.map(orders_df.order_dow)
    df['order_hour_of_day'] = df.order_id.map(orders_df.order_hour_of_day)
    df['days_since_prior_order'] = df.order_id.map(
        orders_df.days_since_prior_order)
    df['days_since_ratio'] = \
        df.days_since_prior_order / df.user_average_days_between_orders

    print('product related features')
    df['aisle_id'] = df.product_id.map(products.aisle_id)
    df['department_id'] = df.product_id.map(products.department_id)
    df['product_orders'] = df.product_id.map(products.orders).astype(np.int32)
    df['product_reorders'] = df.product_id.map(products.reorders)
    df['product_reorder_rate'] = df.product_id.map(products.reorder_rate)

    print('user_X_product related features')
    df['z'] = df.user_id.astype(np.uint64) * 100000 + df.product_id.astype(
        np.uint64)
    # df.drop(['user_id'], axis=1, inplace=True)
    df['UP_orders'] = df.z.map(userXproduct.nb_orders)
    df['UP_orders_ratio'] = (df.UP_orders / df.user_total_orders).astype(
        np.float32)
    df['UP_last_order_id'] = df.z.map(userXproduct.last_order_id)
    df['UP_average_pos_in_cart'] = (
     df.z.map(userXproduct.sum_pos_in_cart) / df.UP_orders).astype(np.float32)
    """df['UP_reorder_rate'] = (df.UP_orders / df.user_total_orders).astype(
        np.float32)"""  # it is the same as UP_orders_ratio
    df['UP_orders_since_last'] = df.user_total_orders - df.UP_last_order_id.map(
        orders_df.order_number)
    df['UP_delta_hour_vs_last'] = abs(
        df.order_hour_of_day - df.UP_last_order_id.map(
            orders_df.order_hour_of_day)).map(lambda x: min(x, 24 - x)).astype(
        np.uint8)
    df['UP_same_dow_as_last_order'] = df.UP_last_order_id.map(
         orders_df.order_dow) == df.order_id.map(orders_df.order_dow)
    df = pd.merge(df, df_order_streak, how='left',
                  on=['product_id', 'user_id'])
    df.drop(['UP_last_order_id', 'z'], axis=1, inplace=True)
    print(df.dtypes)
    print(df.memory_usage())
    return df


def create_products_faron(df):
    products = df.product_id.values
    prob = df.prediction.values

    sort_index = np.argsort(prob)[::-1]
    L2 = products[sort_index]
    P2 = prob[sort_index]

    opt = F1Optimizer.maximize_expectation(P2)

    best_prediction = ['None'] if opt[1] else []
    best_prediction += list(L2[:opt[0]])

    best = ' '.join(map(lambda x: str(x), best_prediction))
    df = df[0:1]
    df.loc[:, 'products'] = best
    return df

if __name__ == '__main__':
    if not os.path.isfile(FEATURE_PATH):
        df_train, df_test = load_n_build_data()
    else:
        print('read HDF5 files..')
        df_train = pd.read_hdf(FEATURE_PATH, 'train')
        df_test = pd.read_hdf(FEATURE_PATH, 'test')

    f_to_use = ['user_total_orders',
                'user_total_items',
                'total_distinct_items',
                'user_average_days_between_orders',
                'user_average_basket',
                'order_hour_of_day',
                'days_since_prior_order',
                'days_since_ratio',
                'aisle_id', 'department_id', 'product_orders',
                'product_reorders',
                'product_reorder_rate',
                'UP_orders',
                'UP_orders_ratio',
                'UP_average_pos_in_cart',
                'UP_orders_since_last',
                'UP_delta_hour_vs_last',
                'dow',
                'UP_same_dow_as_last_order',
                'order_streak']

    """X_train, X_val, y_train, y_val =
                        train_test_split(df_train[f_to_use], labels,
                                                      test_size=0.9,
                                                      random_state=2017)"""

    """
    # split happens in gradientboosting machines themselves
    X_train = df_train[df_train.user_id % 10 != 0]
    X_valid = df_train[df_train.user_id % 10 == 0]

    Y_train = X_train.y
    Y_valid = X_valid.y

    X_train = X_train[f_to_use]
    X_valid = X_valid[f_to_use]
    """
    X_train = df_train[f_to_use]
    Y_train = df_train.y
    df_pred = pd.DataFrame()

    # Catboost
    print('train cat')
    cat = CatBoostClassifier(learning_rate=0.08, iterations=300)
    cat.fit(X_train, Y_train)

    # XGB
    print('train xgb')
    dm_train = xgb.DMatrix(X_train, Y_train)
    dm_test = xgb.DMatrix(df_test[f_to_use])

    watchlist = [(dm_train, "train")]
    bst = xgb.train(params=xgb_params,
                    dtrain=dm_train,
                    num_boost_round=300,
                    evals=watchlist,
                    verbose_eval=10)

    # predict
    print('predict with cat')
    df_pred['cat'] = cat.predict_proba(df_test[f_to_use])[:, 1]
    print('predict with xgb')
    df_pred['xgb'] = bst.predict(dm_test)

    # median of models' prediction
    df_test['prediction'] = df_pred.median(axis=1)
    del df_pred



    """TRESHOLD = 0.22  # guess, should be tuned with crossval on a subset of traindata

    d = {}
    for row in df_test.itertuples():
        if row.pred > TRESHOLD:
            try:
                d[row.order_id] += ' ' + str(row.product_id)
            except:
                d[row.order_id] = str(row.product_id)

    for order in df_test.order_id:
        if order not in d:
            d[order] = 'None'

    sub = pd.DataFrame.from_dict(d, orient='index')
    sub.reset_index(inplace=True)
    sub.columns = ['order_id', 'products']
    sub.to_csv('out/sub_OrderStreak.csv', index=False)
    """
    df_test_filtered = df_test.loc[df_test.prediction > 0.01,
                                   ['order_id', 'prediction', 'product_id']]
    print('Optimize f1..')
    sub = apply_parallel(df_test_filtered.groupby(df_test_filtered.order_id),
                         create_products_faron).reset_index()
    print('save submission')
    sub[['order_id', 'products']].to_csv('out/sub_cat_xgb_f1opt.csv',
                                         index=False)


