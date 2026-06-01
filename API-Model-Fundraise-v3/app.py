import os
import time
import joblib
import requests
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
from google import genai
from google.genai import types
from tensorflow.keras import layers
from prometheus_fastapi_instrumentator import Instrumentator
from ai_service import get_financial_summary

load_dotenv()

UMKM_MASTER_CACHE = None
CACHE_EXPIRATION = 0 
CACHE_DURATION = 7200
BACKEND_URL = os.getenv("BACKEND_URL")
API_KEY = os.getenv("API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

app = FastAPI(title="FundRaise ML API - PRODUCTION")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Instrumentator().instrument(app).expose(app)

# Custom Layers & Model FT-Transformer
@tf.keras.utils.register_keras_serializable()
class FeatureTokenizer(layers.Layer):
    def __init__(self, n_features, d_token, **kwargs):
        super().__init__(**kwargs)
        self.n_features = n_features
        self.d_token = d_token

    def build(self, input_shape):
        self.weight = self.add_weight(
            shape=(self.n_features, self.d_token),
            initializer="glorot_uniform", trainable=True, name="feature_weight"
        )
        self.bias = self.add_weight(
            shape=(self.n_features, self.d_token),
            initializer="zeros", trainable=True, name="feature_bias"
        )
        self.cls_token = self.add_weight(
            shape=(1, 1, self.d_token),
            initializer="random_normal", trainable=True, name="cls_token"
        )

    def get_config(self):
        config = super().get_config()
        config.update({"n_features": self.n_features, "d_token": self.d_token})
        return config

    def call(self, x):
        x = tf.expand_dims(x, axis=-1)
        tokens = x * self.weight + self.bias
        cls = tf.repeat(self.cls_token, repeats=tf.shape(x)[0], axis=0)
        return tf.concat([cls, tokens], axis=1)
    
@tf.keras.utils.register_keras_serializable()
class TransformerBlock(layers.Layer):
    def __init__(self, d_token, n_heads, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_token = d_token
        self.n_heads = n_heads
        self.dropout_rate = dropout
        self.norm1 = layers.LayerNormalization()
        self.attention = layers.MultiHeadAttention(num_heads=n_heads, key_dim=d_token // n_heads, dropout=dropout)
        self.dropout1 = layers.Dropout(dropout)
        self.norm2 = layers.LayerNormalization()
        self.ffn = tf.keras.Sequential([
            layers.Dense(d_token * 2, activation="gelu"),
            layers.Dropout(dropout),
            layers.Dense(d_token)
        ])
        self.dropout2 = layers.Dropout(dropout)

    def get_config(self):
        config = super().get_config()
        config.update({"d_token": self.d_token, "n_heads": self.n_heads, "dropout": self.dropout_rate})
        return config

    def call(self, x, training=False):
        attn_output = self.attention(self.norm1(x), self.norm1(x), training=training)
        x = x + self.dropout1(attn_output, training=training)
        ffn_output = self.ffn(self.norm2(x), training=training)
        return x + self.dropout2(ffn_output, training=training)

@tf.keras.utils.register_keras_serializable()
class FTTransformer(tf.keras.Model):
    def __init__(self, n_features, n_classes, d_token=64, n_heads=4, n_blocks=3, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.n_features = n_features
        self.n_classes = n_classes
        self.d_token = d_token
        self.n_heads = n_heads
        self.n_blocks = n_blocks
        self.dropout_rate = dropout
        self.tokenizer_layer = FeatureTokenizer(n_features, d_token)
        self.transformer_blocks = [TransformerBlock(d_token, n_heads, dropout) for _ in range(n_blocks)]
        self.final_norm = layers.LayerNormalization()
        self.classifier = tf.keras.Sequential([
            layers.Dense(128, activation="relu"),
            layers.Dropout(dropout),
            layers.Dense(n_classes, activation="softmax")
        ])
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "n_features": self.n_features, "n_classes": self.n_classes,
            "d_token": self.d_token, "n_heads": self.n_heads,
            "n_blocks": self.n_blocks, "dropout": self.dropout_rate,
        })
        return config

    def call(self, x, training=False):
        x = self.tokenizer_layer(x)
        for block in self.transformer_blocks: x = block(x, training=training)
        cls_token = self.final_norm(x[:, 0])
        return self.classifier(cls_token, training=training)

# Load Models & Pipelines
bundle_investor = joblib.load("pipeline_investor.joblib")
preprocessor_investor = bundle_investor["preprocessor"]
inverse_mapping_investor = bundle_investor["inverse_mapping"]
bundle_model = joblib.load("PolyEns.joblib")
preprocessor_tambahan_investor = bundle_model["poly_features"]
best_model_ensemble = bundle_model["ensemble_model"]

bundle_umkm = joblib.load("pipeline.joblib")
preprocessor_umkm = bundle_umkm["preprocessor"]
inverse_mapping_umkm = bundle_umkm["inverse_mapping"]
ft_model = tf.keras.models.load_model(
    "best_ft_transformer.keras",
    custom_objects={
        "FTTransformer": FTTransformer,
        "FeatureTokenizer": FeatureTokenizer,
        "TransformerBlock": TransformerBlock
    }
)

# Skema Input
class UMKMSurvey(BaseModel):
    net_profit_margin: float
    kepuasan_pelanggan: float
    peak_hour_latency: str
    review_volatility: float
    repeat_order_rate: float
    digital_adoption_score: float
    year_revenue: int
    business_tenure_years: float

class InvestorSurvey(BaseModel):
    investor_id: int 
    kepuasan_pelanggan: float
    peak_hour_latency: str = "medium"
    digital_adoption_score: float
    net_profit_margin: float
    year_revenue: int
    business_tenure_years: float


# Endpoint Klasifikasi UMKM
@app.post("/classify-umkm")
def classify_umkm(survey: UMKMSurvey):
    df_umkm = pd.DataFrame([survey.dict()])[
        ['net_profit_margin', 'kepuasan_pelanggan', 'peak_hour_latency', 'review_volatility', 
         'repeat_order_rate', 'digital_adoption_score', 'year_revenue', 'business_tenure_years']
    ]
    X_Processed = preprocessor_umkm.transform(df_umkm)
    Prob = ft_model.predict(X_Processed)
    pred_class_num = int(np.argmax(Prob[0]))
    
    return {
        "status": "success",
        "predicted_class_id": pred_class_num,
        "predicted_class_label": inverse_mapping_investor[pred_class_num],
    }

# Endpoint Rekomendasi
@app.post("/recommend")
def get_recommendations(survey: InvestorSurvey):
    global UMKM_MASTER_CACHE, CACHE_EXPIRATION
    
    FEATURES = [
        'kepuasan_pelanggan', 'digital_adoption_score', 'net_profit_margin', 
        'year_revenue', 'business_tenure_years', 'peak_hour_latency'
    ]
    
    df_investor = pd.DataFrame([survey.dict()])[FEATURES]
    X_investor_base = preprocessor_investor.transform(df_investor)
    df_for_poly = pd.DataFrame(X_investor_base, columns=FEATURES)
    X_for_model = preprocessor_tambahan_investor.transform(df_for_poly)
    
    pred_class_num = best_model_ensemble.predict(X_for_model)[0]
    pred_class_label = inverse_mapping_investor[pred_class_num]
    
    current_time = time.time()
    if UMKM_MASTER_CACHE is None or current_time > CACHE_EXPIRATION:
        headers = {"x-api-key": API_KEY} 
        try:
            response = requests.get(BACKEND_URL, headers=headers, timeout=5)
            response.raise_for_status()
            UMKM_MASTER_CACHE = response.json().get("data") or []
            CACHE_EXPIRATION = current_time + CACHE_DURATION
        except Exception as e:
            if UMKM_MASTER_CACHE is None:
                raise HTTPException(status_code=503, detail=f"Backend Unreachable: {str(e)}")

    umkm_list = UMKM_MASTER_CACHE
    filtered_umkm = [
        u for u in umkm_list 
        if u is not None and u.get("class") is not None and int(u.get("class")) == pred_class_num
    ]
    
    recommendations = []
    float_cols = ['kepuasan_pelanggan', 'digital_adoption_score', 'net_profit_margin', 'year_revenue', 'business_tenure_years']
    
    if len(filtered_umkm) > 0:
        df_umkm_batch = pd.DataFrame(filtered_umkm)[FEATURES]
        df_umkm_batch[float_cols] = df_umkm_batch[float_cols].astype(float)
        
        X_umkm_processed_batch = preprocessor_investor.transform(df_umkm_batch)
        
        sim_scores_array = cosine_similarity(X_investor_base, X_umkm_processed_batch)[0]
        sim_scores_array = np.maximum(sim_scores_array, 0)
        skor_persen_array = np.round(sim_scores_array * 100, 2)

        recommendations = [
            {
                "id_bisnis": umkm['bisnis']['id'] if 'bisnis' in umkm else "N/A",
                "skor_kecocokan": float(skor_persen_array[i]),
                "matched_class": pred_class_label
            }
            for i, umkm in enumerate(filtered_umkm)
        ]
            
    return {"results": sorted(recommendations, key=lambda x: x['skor_kecocokan'], reverse=True)}

@app.post("/generate-financial-summary")
def generate_summary_endpoint(survey: UMKMSurvey):
    try:
        # Panggil fungsi AI dengan data dari request body
        summary = get_financial_summary(survey.dict())
        return {"status": "success", "summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    models_status = {
        "ensemble_investor": best_model_ensemble is not None,
        "ft_transformer_umkm": ft_model is not None,
        "preprocessor_investor": preprocessor_investor is not None,
        "preprocessor_umkm": preprocessor_umkm is not None
    }
    
    if not all(models_status.values()):
        raise HTTPException(
            status_code=503, 
            detail={
                "status": "error",
                "message": "API is running, but some ML models failed to load.",
                "models": models_status
            }
        )
    
    return {
        "status": "success",
        "message": "FundRaise ML API is healthy and ready to rock! 🚀",
        "models_loaded": True
    }