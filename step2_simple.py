# step2_simple.py - Basit İkinci Adım

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_data():
    """Veri yükle"""
    ratings_df = pd.read_csv('data/processed/ratings.csv')
    movies_df = pd.read_csv('data/processed/movies.csv')
    users_df = pd.read_csv('data/processed/users.csv')
    print("✅ Veriler yüklendi")
    return ratings_df, movies_df, users_df

def basic_analysis(ratings_df, movies_df, users_df):
    """Temel analiz"""
    print("\n📊 TEMEL İSTATİSTİKLER:")
    
    # Sparsity hesapla
    total_possible = len(users_df) * len(movies_df)
    actual_ratings = len(ratings_df)
    sparsity = (1 - actual_ratings / total_possible) * 100
    print(f"Sparsity: {sparsity:.1f}% (veri ne kadar boş)")
    
    # Kullanıcı analizi
    user_counts = ratings_df.groupby('user_id').size()
    print(f"Ortalama kullanıcı aktivitesi: {user_counts.mean():.1f} rating")
    print(f"En aktif kullanıcı: {user_counts.max()} rating")
    
    # Film analizi  
    movie_counts = ratings_df.groupby('movie_id').size()
    print(f"Ortalama film popülerliği: {movie_counts.mean():.1f} rating")
    print(f"En popüler film: {movie_counts.max()} rating")
    
    # Cold start
    single_users = (user_counts == 1).sum()
    single_movies = (movie_counts == 1).sum()
    print(f"Cold start users: {single_users}")
    print(f"Cold start movies: {single_movies}")

def create_user_item_matrix(ratings_df):
    """User-Item matrix oluştur"""
    print("\n🔄 User-Item matrix oluşturuluyor...")
    
    # Pivot table oluştur
    user_item_matrix = ratings_df.pivot_table(
        index='user_id', 
        columns='movie_id', 
        values='rating', 
        fill_value=0
    )
    
    print(f"Matrix boyutu: {user_item_matrix.shape}")
    print(f"Toplam rating sayısı: {(user_item_matrix != 0).sum().sum()}")
    
    # Matrix'i kaydet
    user_item_matrix.to_csv('data/processed/user_item_matrix.csv')
    print("✅ User-item matrix kaydedildi: data/processed/user_item_matrix.csv")
    
    return user_item_matrix

def prepare_train_test_split(ratings_df):
    """Train-test ayrımı hazırla"""
    print("\n✂️ Train-Test ayrımı yapılıyor...")
    
    from sklearn.model_selection import train_test_split
    
    # %80 train, %20 test
    train_df, test_df = train_test_split(
        ratings_df, 
        test_size=0.2, 
        random_state=42,
        stratify=ratings_df['rating']  # Rating dağılımını koru
    )
    
    print(f"Train set: {len(train_df):,} rating")
    print(f"Test set: {len(test_df):,} rating")
    
    # Kaydet
    train_df.to_csv('data/processed/train_ratings.csv', index=False)
    test_df.to_csv('data/processed/test_ratings.csv', index=False)
    
    print("✅ Train-test setleri kaydedildi")
    return train_df, test_df

def simple_visualization(ratings_df):
    """Basit görselleştirme"""
    print("\n📈 Basit grafik oluşturuluyor...")
    
    # Tek grafik
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Film Öneri Sistemi - Basit Analiz', fontsize=14, fontweight='bold')
    
    # 1. Rating dağılımı
    ratings_df['rating'].value_counts().sort_index().plot(kind='bar', ax=ax1, color='skyblue')
    ax1.set_title('Rating Dağılımı')
    ax1.set_ylabel('Sayı')
    
    # 2. Kullanıcı aktivitesi
    user_activity = ratings_df.groupby('user_id').size()
    user_activity.hist(bins=30, ax=ax2, color='lightgreen', alpha=0.7)
    ax2.set_title('Kullanıcı Aktivitesi')
    ax2.set_xlabel('Rating sayısı')
    
    # 3. Film popülerliği
    movie_popularity = ratings_df.groupby('movie_id').size()
    movie_popularity.hist(bins=50, ax=ax3, color='salmon', alpha=0.7)
    ax3.set_title('Film Popülerliği')
    ax3.set_xlabel('Rating sayısı')
    
    # 4. Rating ortalamaları
    movie_avg_ratings = ratings_df.groupby('movie_id')['rating'].mean()
    movie_avg_ratings.hist(bins=20, ax=ax4, color='gold', alpha=0.7)
    ax4.set_title('Film Rating Ortalamaları')
    ax4.set_xlabel('Ortalama Rating')
    
    plt.tight_layout()
    plt.savefig('results/figures/simple_analysis.png', dpi=150, bbox_inches='tight')
    print("📊 Grafik kaydedildi: results/figures/simple_analysis.png")
    plt.show()

def main():
    """Ana fonksiyon"""
    print("🎯 BASİT İKİNCİ ADIM BAŞLIYOR")
    print("="*40)
    
    # 1. Veri yükle
    ratings_df, movies_df, users_df = load_data()
    
    # 2. Temel analiz
    basic_analysis(ratings_df, movies_df, users_df)
    
    # 3. User-Item matrix
    user_item_matrix = create_user_item_matrix(ratings_df)
    
    # 4. Train-Test split
    train_df, test_df = prepare_train_test_split(ratings_df)
    
    # 5. Basit görselleştirme
    simple_visualization(ratings_df)
    
    print("\n✅ BASİT İKİNCİ ADIM TAMAMLANDI!")
    print("🔜 Sonraki adım: Model geliştirme")

if __name__ == "__main__":
    main()