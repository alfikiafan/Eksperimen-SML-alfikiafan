import pandas as pd
import os
import joblib
import logging
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def load_data(path):
    if not os.path.exists(path):
        logger.error(f"Dataset not found at path: {path}")
        raise FileNotFoundError(f"Dataset not found at {path}")
    df = pd.read_csv(path)
    logger.info(f"Dataset loaded with shape: {df.shape}")
    return df

def preprocess_data(df):
    logger.info("Handling missing values...")
    df.fillna({
        'Time_spent_Alone': df['Time_spent_Alone'].median(),
        'Social_event_attendance': df['Social_event_attendance'].median(),
        'Going_outside': df['Going_outside'].median(),
        'Friends_circle_size': df['Friends_circle_size'].median(),
        'Post_frequency': df['Post_frequency'].median(),
        'Stage_fear': df['Stage_fear'].mode()[0],
        'Drained_after_socializing': df['Drained_after_socializing'].mode()[0]
    }, inplace=True)

    logger.info("Encoding categorical variables...")
    le = LabelEncoder()
    df['Stage_fear'] = LabelEncoder().fit_transform(df['Stage_fear'])
    df['Drained_after_socializing'] = LabelEncoder().fit_transform(df['Drained_after_socializing'])
    df['Personality'] = LabelEncoder().fit_transform(df['Personality'])

    logger.info("Scaling numerical features...")
    numerical_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    logger.info("Splitting dataset...")
    X = df.drop('Personality', axis=1)
    y = df['Personality']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, scaler, numerical_cols

def save_artifacts(X_train, X_test, y_train, y_test, scaler, feature_names, output_dir="preprocessing/personality_dataset_preprocessing"):
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Saving artifacts to: {output_dir}")

    joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))
    joblib.dump(feature_names, os.path.join(output_dir, 'feature_names.pkl'))

    train_df = pd.concat([X_train, y_train.reset_index(drop=True)], axis=1)
    test_df = pd.concat([X_test, y_test.reset_index(drop=True)], axis=1)

    train_df.to_csv(os.path.join(output_dir, 'train_data.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test_data.csv'), index=False)

    logger.info("Train and test datasets saved successfully.")


def main():
    try:
        logger.info("Starting preprocessing pipeline...")
        data_path = "../personality_dataset_raw.csv"
        df = load_data(data_path)
        X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_data(df)
        save_artifacts(X_train, X_test, y_train, y_test, scaler, feature_names)
        logger.info("Pipeline finished successfully.")
    except Exception as e:
        logger.exception("An error occurred during preprocessing.")
        raise e

if __name__ == "__main__":
    main()
