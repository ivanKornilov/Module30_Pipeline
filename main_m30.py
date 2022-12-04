import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def main():
    print('Задача многоклассовой классификации, предсказать категорию цены (high, medium или low), за которую можно продать автомобиль')
    df = pd.read_csv('data/homework.csv')

    def filter_data(df):
        columns_to_drop = [
            'id',
            'url',
            'region',
            'region_url',
            'price',
            'manufacturer',
            'image_url',
            'description',
            'posting_date',
            'lat',
            'long'
        ]
        df.drop(columns_to_drop, axis=1)
        return df

    def filter_boundaries(df):
        def calculate_outliers(data):
            q25 = data.quantile(0.25)
            q75 = data.quantile(0.75)
            iqr = q75 - q25
            boundaries = (q25 - 1.5 * iqr, q75 + 1.5 * iqr)
            return boundaries
        boundaries = calculate_outliers(df['year'])
        df.loc[df['year'] < boundaries[0], 'year'] = round(boundaries[0])
        df.loc[df['year'] > boundaries[1], 'year'] = round(boundaries[1])
        return df

    def filter_short_model(df):
        def short_model(x):
            if not pd.isna(x):
                return x.lower().split(' ')[0]
            else:
                return x
        df.loc[:, 'short_model'] = df['model'].apply(short_model)
        df.loc[:, 'age_category'] = df['year'].apply(lambda x: 'new' if x > 2013 else ('old' if x < 2006 else 'average'))
        return df

    # Копируем для модели Х - данные для прогнозирования, у - целевая переменная
    X = df.drop(['price_category'], axis=1)
    y = df['price_category']

    numerical = X.select_dtypes(include=['int64', 'float64']).columns
    categorical = X.select_dtypes(include=['object']).columns

    numerical_transfromer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('numerical', numerical_transfromer, numerical),
        ('categirical', categorical_transformer, categorical)
    ])

    models = (
        LogisticRegression(solver='liblinear'),
        RandomForestClassifier(),
        MLPClassifier(activation='logistic', hidden_layer_sizes=(256, 128, 64))
    )

    best_score = .0
    best_pipe = None
    for model in models:
        pipe = Pipeline(steps=[
            ('filter', FunctionTransformer(filter_boundaries)),
            ('filter2', FunctionTransformer(filter_short_model)),
            ('filter3', FunctionTransformer(filter_data)),
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        score = cross_val_score(pipe, X, y, cv=4, scoring='accuracy')
        print(f'model: {type(model).__name__}, acc_mean: {score.mean():.4f}, acc_std: {score.std():.4f}')
        if score.mean() > best_score:
            best_score = score.mean()
            best_pipe = pipe
    print(f'Best model: {type(best_pipe.named_steps["classifier"]).__name__}, accuracy: {best_score:.4f}')
    joblib.dump(best_pipe, 'loan_pipe.pkl')


if __name__ == '__main__':
    main()
