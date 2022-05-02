import pandas as pd
from tensorflow import keras
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from mlrun.frameworks.auto_mlrun.auto_mlrun import AutoMLRun

def get_data():
    X, y = load_iris(return_X_y=True, as_frame=True)
    y_one_hot = pd.get_dummies(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_one_hot, test_size=0.33, random_state=42, stratify=y_one_hot
    )
    
    return X_train, X_test, y_train, y_test

def log_test_set(context):
    X_train, X_test, y_train, y_test = get_data()
    context.log_dataset("X_test", X_test, index=False, format="csv")
    context.log_dataset("y_test", y_test, index=False, format="csv")

def train_reference(context):
    X_train, X_test, y_train, y_test = get_data()
    
    model = keras.Sequential([
        keras.layers.Dense(10, activation='relu'),
        keras.layers.Dense(10, activation='relu'),
        keras.layers.Dense(3, activation='softmax')
      ])
    
    AutoMLRun.apply_mlrun(
        model=model,
        model_name="reference",
        context=context,
    )

    model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.RMSprop(),
        metrics=['accuracy']
    )
    
    model.fit(
        x=X_train,
        y=y_train,
        validation_data=(X_test, y_test),
        batch_size=128,
        epochs=5,
        verbose=1,
    )

def train_current(context):
    X_train, X_test, y_train, y_test = get_data()
    
    model = keras.Sequential([
        keras.layers.Dense(10, activation='relu'),
        keras.layers.Dense(20, activation='relu'),
        keras.layers.Dense(10, activation='relu'),
        keras.layers.Dense(3, activation='softmax')
      ])
    
    AutoMLRun.apply_mlrun(
        model=model,
        model_name="current",
        context=context,
    )

    model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.RMSprop(),
        metrics=['accuracy']
    )
    
    model.fit(
        x=X_train,
        y=y_train,
        validation_data=(X_test, y_test),
        batch_size=128,
        epochs=5,
        verbose=1,
    )