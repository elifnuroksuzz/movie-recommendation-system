# main.py - Film Öneri Sistemi İlk Adım

import os
import requests
import zipfile
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Rastgelelik için seed belirleme (reproducibility)
np.random.seed(42)

def create_project_structure():
    """Proje klasör yapısını oluşturur"""
    directories = [
        'data/raw',
        'data/processed', 
        'data/external',
        'src',
        'notebooks',
        'models',
        'results/figures',
        'results/metrics',
        'results/reports'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Klasör oluşturuldu: {directory}")

def download_movielens_data():
    """MovieLens 100K veri setini indirir ve çıkarır"""
    url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
    zip_path = "data/raw/ml-100k.zip"
    
    print("📥 MovieLens 100K veri seti indiriliyor...")
    
    try:
        # Veri setini indir
        response = requests.get(url)
        response.raise_for_status()
        
        with open(zip_path, 'wb') as f:
            f.write(response.content)
        
        # ZIP dosyasını çıkar
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall("data/raw/")
        
        print("✅ Veri seti başarıyla indirildi ve çıkarıldı!")
        return "data/raw/ml-100k/"
        
    except Exception as e:
        print(f"❌ Veri indirme hatası: {e}")
        return None

def load_movielens_data(data_path):
    """MovieLens veri setini yükler"""
    
    try:
        # Ratings verisi (u.data)
        ratings_columns = ['user_id', 'movie_id', 'rating', 'timestamp']
        ratings_df = pd.read_csv(
            os.path.join(data_path, 'u.data'), 
            sep='\t', 
            names=ratings_columns,
            encoding='latin-1'
        )
        
        # Film bilgileri (u.item)
        movies_columns = [
            'movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url'
        ] + [f'genre_{i}' for i in range(19)]
        
        movies_df = pd.read_csv(
            os.path.join(data_path, 'u.item'), 
            sep='|', 
            names=movies_columns,
            encoding='latin-1'
        )
        
        # Kullanıcı bilgileri (u.user)
        users_columns = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
        users_df = pd.read_csv(
            os.path.join(data_path, 'u.user'), 
            sep='|', 
            names=users_columns,
            encoding='latin-1'
        )
        
        print("✅ Veri setleri başarıyla yüklendi!")
        print(f"📊 Ratings: {len(ratings_df):,} kayıt")
        print(f"🎬 Filmler: {len(movies_df):,} film") 
        print(f"👥 Kullanıcılar: {len(users_df):,} kullanıcı")
        
        return ratings_df, movies_df, users_df
        
    except Exception as e:
        print(f"❌ Veri yükleme hatası: {e}")
        return None, None, None

def basic_data_exploration(ratings_df, movies_df, users_df):
    """Temel veri keşfi ve istatistikler"""
    print("\n" + "="*50)
    print("📈 TEMEL VERİ İSTATİSTİKLERİ")
    print("="*50)
    
    # Ratings istatistikleri
    print("\n🌟 Rating Dağılımı:")
    rating_counts = ratings_df['rating'].value_counts().sort_index()
    for rating, count in rating_counts.items():
        print(f"   {rating} yıldız: {count:,} rating")
    
    print(f"\n📊 Ortalama Rating: {ratings_df['rating'].mean():.2f}")
    print(f"📊 Rating Standart Sapması: {ratings_df['rating'].std():.2f}")
    
    # Kullanıcı aktivitesi
    user_activity = ratings_df.groupby('user_id')['rating'].count()
    print(f"\n👤 Kullanıcı başına ortalama rating sayısı: {user_activity.mean():.2f}")
    print(f"👤 En aktif kullanıcı: {user_activity.max():,} rating")
    print(f"👤 En az aktif kullanıcı: {user_activity.min():,} rating")
    
    # Film popülaritesi
    movie_popularity = ratings_df.groupby('movie_id')['rating'].count()
    print(f"\n🎬 Film başına ortalama rating sayısı: {movie_popularity.mean():.2f}")
    print(f"🎬 En popüler film: {movie_popularity.max():,} rating")
    print(f"🎬 En az popüler film: {movie_popularity.min():,} rating")
    
    # En popüler filmleri göster
    popular_movies = ratings_df.groupby('movie_id').agg({
        'rating': ['count', 'mean']
    }).round(2)
    popular_movies.columns = ['rating_count', 'rating_avg']
    popular_movies = popular_movies.reset_index()
    
    # Film isimlerini ekle
    popular_movies = popular_movies.merge(
        movies_df[['movie_id', 'title']], on='movie_id'
    )
    
    # En çok rating alan 10 film
    top_rated_movies = popular_movies.nlargest(10, 'rating_count')
    print(f"\n🏆 EN POPÜLER 10 FİLM:")
    for idx, row in top_rated_movies.iterrows():
        print(f"   {row['title'][:50]}: {row['rating_count']} rating (ort: {row['rating_avg']:.2f})")

def save_processed_data(ratings_df, movies_df, users_df):
    """İşlenmiş veriyi kaydet"""
    try:
        ratings_df.to_csv('data/processed/ratings.csv', index=False)
        movies_df.to_csv('data/processed/movies.csv', index=False)
        users_df.to_csv('data/processed/users.csv', index=False)
        print("\n💾 İşlenmiş veriler kaydedildi:")
        print("   📁 data/processed/ratings.csv")
        print("   📁 data/processed/movies.csv") 
        print("   📁 data/processed/users.csv")
        return True
    except Exception as e:
        print(f"❌ Veri kaydetme hatası: {e}")
        return False

def main():
    """Ana çalıştırma fonksiyonu"""
    print("🚀 FILM ÖNERİ SİSTEMİ PROJESİ BAŞLATILIYOR")
    print("="*60)
    
    # 1. Proje yapısını oluştur
    print("\n📁 Proje klasör yapısı oluşturuluyor...")
    create_project_structure()
    
    # 2. Veri setini indir
    print("\n📥 Veri seti indiriliyor...")
    data_path = download_movielens_data()
    
    if data_path is None:
        print("❌ Veri indirilemedi, proje durduruluyor.")
        return
    
    # 3. Veri setlerini yükle
    print("\n📊 Veri setleri yükleniyor...")
    ratings_df, movies_df, users_df = load_movielens_data(data_path)
    
    if ratings_df is None:
        print("❌ Veri yüklenemedi, proje durduruluyor.")
        return
    
    # 4. Temel keşifsel analiz
    basic_data_exploration(ratings_df, movies_df, users_df)
    
    # 5. İşlenmiş veriyi kaydet
    print("\n💾 Veriler kaydediliyor...")
    save_success = save_processed_data(ratings_df, movies_df, users_df)
    
    if save_success:
        print("\n✅ İLK ADIM BAŞARIYLA TAMAMLANDI!")
        print("\n🔜 Sonraki adım için hazır!")
        print("   - Detaylı veri analizi")
        print("   - Görselleştirmeler") 
        print("   - Model geliştirme")
    else:
        print("\n⚠️ Bazı hatalar oluştu, lütfen kontrol edin.")

# Programı çalıştır
if __name__ == "__main__":
    main()