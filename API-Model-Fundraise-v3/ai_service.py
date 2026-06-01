import os
from dotenv import load_dotenv
from google import genai 
from google.genai import types

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

def get_financial_summary(data_dict):
    try:
        # Inisialisasi client langsung dengan API Key
        client = genai.Client(api_key=GEMINI_API_KEY)
        
        prompt = f"""
        Anda adalah seorang analis keuangan profesional. 
        Berikan ringkasan performa bisnis UMKM dalam 2-3 kalimat singkat untuk investor berdasarkan data berikut:
        - Pendapatan Tahunan: Rp{data_dict['year_revenue']:,}
        - Margin Keuntungan Bersih: {data_dict['net_profit_margin']}%
        - Kepuasan Pelanggan: {data_dict['kepuasan_pelanggan']}/5
        - Skor Adopsi Digital: {data_dict['digital_adoption_score']}%
        - Lama Berusaha: {data_dict['business_tenure_years']} tahun
        
        Berikan analisis apakah bisnis ini sehat dan apa keunggulan utamanya.
        """
        
        response = client.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents=prompt
        )
        
        return response.text.strip()
    
    except Exception as e:
        return f"Analisis AI sementara tidak tersedia: {str(e)}"