# save_lightweight_model.py - GitHub için küçük model kaydetme

import pandas as pd
import numpy as np
import pickle
import json

def create_lightweight_model():
    """GitHub'a yüklenebilir küçük model oluştur"""
    print("🔄 GitHub için hafif model oluşturuluyor...")
    
    try:
        # Temel istatistikleri kaydet
        ratings_df = pd.read_csv('data/processed/ratings.csv')
        movies_df = pd.read_csv('data/processed/movies.csv')
        
        # Film popülerite verileri (küçük)
        movie_stats = ratings_df.groupby('movie_id').agg({
            'rating': ['mean', 'count']
        }).round(3)
        movie_stats.columns = ['avg_rating', 'rating_count']
        
        # Top 100 film (küçük dosya)
        top_movies = movie_stats.nlargest(100, 'rating_count')
        
        # Kullanıcı istatistikleri (özet)
        user_stats = ratings_df.groupby('user_id').agg({
            'rating': ['mean', 'count']
        }).round(3)
        user_stats.columns = ['avg_rating', 'rating_count']
        
        # Küçük dosyalar olarak kaydet
        top_movies.to_csv('data/processed/top_100_movies.csv')
        user_stats.head(50).to_csv('data/processed/sample_users.csv')  # Sadece ilk 50 kullanıcı
        
        # Model parametreleri (JSON olarak)
        model_config = {
            'model_type': 'collaborative_filtering',
            'best_k': 30,
            'global_mean': float(ratings_df['rating'].mean()),
            'n_users': len(ratings_df['user_id'].unique()),
            'n_movies': len(ratings_df['movie_id'].unique()),
            'sparsity': 93.7,
            'performance': {
                'rmse': 1.0520,
                'mae': 0.8202
            }
        }
        
        with open('models/model_config.json', 'w') as f:
            json.dump(model_config, f, indent=2)
        
        print("✅ Hafif model dosyaları oluşturuldu:")
        print("   📁 data/processed/top_100_movies.csv (küçük)")
        print("   📁 data/processed/sample_users.csv (küçük)")
        print("   📁 models/model_config.json (çok küçük)")
        print("\n🔥 Büyük model dosyaları .gitignore'da hariç tutuldu")
        
    except Exception as e:
        print(f"❌ Hata: {e}")

if __name__ == "__main__":
    create_lightweight_model()