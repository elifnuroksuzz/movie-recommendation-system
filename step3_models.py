# step3_models.py - Üçüncü Adım: Model Geliştirme

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle
import warnings
warnings.filterwarnings('ignore')

class CollaborativeFiltering:
    """Kullanıcı-tabanlı Collaborative Filtering"""
    
    def __init__(self, k_users=50):
        self.k_users = k_users  # En benzer k kullanıcı
        self.user_similarity = None
        self.user_item_matrix = None
        self.user_means = None
        
    def fit(self, user_item_matrix):
        """Modeli eğit"""
        print(f"🤝 Collaborative Filtering eğitiliyor... (k={self.k_users})")
        
        self.user_item_matrix = user_item_matrix
        
        # Kullanıcı ortalamalarını hesapla (sadece 0 olmayanlar için)
        self.user_means = {}
        for user in user_item_matrix.index:
            user_ratings = user_item_matrix.loc[user]
            non_zero_ratings = user_ratings[user_ratings > 0]
            self.user_means[user] = non_zero_ratings.mean() if len(non_zero_ratings) > 0 else 0
        
        # Kullanıcı benzerliğini hesapla (cosine similarity)
        print("   📊 Kullanıcı benzerlik matrisi hesaplanıyor...")
        
        # Matrix'i normalize et (kullanıcı ortalamasını çıkar)
        normalized_matrix = user_item_matrix.copy()
        for user in user_item_matrix.index:
            user_mean = self.user_means[user]
            # Sadece rating verilmiş filmleri normalize et
            mask = user_item_matrix.loc[user] > 0
            normalized_matrix.loc[user, mask] = user_item_matrix.loc[user, mask] - user_mean
        
        # Cosine similarity hesapla
        self.user_similarity = cosine_similarity(normalized_matrix.fillna(0))
        
        print("✅ Model eğitimi tamamlandı!")
        
    def predict(self, user_id, movie_id):
        """Tek tahmin yap"""
        if user_id not in self.user_item_matrix.index:
            return 3.0  # Varsayılan değer
        
        if movie_id not in self.user_item_matrix.columns:
            return 3.0  # Varsayılan değer
            
        # Kullanıcının indeksini bul
        user_idx = list(self.user_item_matrix.index).index(user_id)
        
        # Bu kullanıcıya en benzer kullanıcıları bul
        user_similarities = self.user_similarity[user_idx]
        
        # Kendisini hariç tut
        user_similarities[user_idx] = -1
        
        # En benzer k kullanıcıyı seç
        similar_users_idx = np.argsort(user_similarities)[-self.k_users:]
        
        weighted_sum = 0
        similarity_sum = 0
        
        for similar_user_idx in similar_users_idx:
            if user_similarities[similar_user_idx] <= 0:
                continue
                
            similar_user_id = self.user_item_matrix.index[similar_user_idx]
            similar_user_rating = self.user_item_matrix.loc[similar_user_id, movie_id]
            
            if similar_user_rating > 0:  # Bu kullanıcı bu filme rating vermiş
                similarity = user_similarities[similar_user_idx]
                similar_user_mean = self.user_means[similar_user_id]
                
                weighted_sum += similarity * (similar_user_rating - similar_user_mean)
                similarity_sum += similarity
        
        if similarity_sum == 0:
            return self.user_means[user_id]
        
        prediction = self.user_means[user_id] + (weighted_sum / similarity_sum)
        
        # Rating aralığında tut (1-5)
        return max(1, min(5, prediction))
    
    def predict_batch(self, test_pairs):
        """Toplu tahmin"""
        predictions = []
        print(f"🔮 {len(test_pairs)} tahmin yapılıyor...")
        
        for i, (user_id, movie_id) in enumerate(test_pairs):
            pred = self.predict(user_id, movie_id)
            predictions.append(pred)
            
            if (i + 1) % 5000 == 0:
                print(f"   {i + 1}/{len(test_pairs)} tahmin tamamlandı")
        
        return np.array(predictions)

class ContentBasedFiltering:
    """İçerik-tabanlı Filtering"""
    
    def __init__(self):
        self.item_similarity = None
        self.movies_df = None
        self.user_profiles = None
        
    def fit(self, movies_df, ratings_df):
        """Modeli eğit"""
        print("🎬 Content-Based Filtering eğitiliyor...")
        
        self.movies_df = movies_df
        
        # Genre özelliklerini al
        genre_features = movies_df[[col for col in movies_df.columns if col.startswith('genre_')]]
        
        # Film benzerliğini hesapla
        print("   🎭 Film benzerlik matrisi hesaplanıyor...")
        self.item_similarity = cosine_similarity(genre_features)
        
        # Kullanıcı profillerini oluştur (hangi türleri seviyor)
        print("   👤 Kullanıcı profilleri oluşturuluyor...")
        self.user_profiles = {}
        
        for user_id in ratings_df['user_id'].unique():
            user_ratings = ratings_df[ratings_df['user_id'] == user_id]
            
            # Bu kullanıcının verdiği yüksek rating'li filmler (4-5)
            liked_movies = user_ratings[user_ratings['rating'] >= 4]['movie_id'].values
            
            # Bu filmlerin genre özelliklerini topla
            user_profile = np.zeros(len(genre_features.columns))
            
            for movie_id in liked_movies:
                if movie_id in movies_df['movie_id'].values:
                    movie_idx = movies_df[movies_df['movie_id'] == movie_id].index[0]
                    user_profile += genre_features.iloc[movie_idx].values
            
            # Normalize et
            if np.sum(user_profile) > 0:
                user_profile = user_profile / np.sum(user_profile)
            
            self.user_profiles[user_id] = user_profile
        
        print("✅ Content-Based model eğitimi tamamlandı!")
    
    def predict(self, user_id, movie_id):
        """Tek tahmin yap"""
        if user_id not in self.user_profiles:
            return 3.0
            
        if movie_id not in self.movies_df['movie_id'].values:
            return 3.0
        
        # Film özelliklerini al
        movie_idx = self.movies_df[self.movies_df['movie_id'] == movie_id].index[0]
        movie_features = self.movies_df[[col for col in self.movies_df.columns if col.startswith('genre_')]].iloc[movie_idx].values
        
        # Kullanıcı profili ile film özelliği arasındaki benzerlik
        user_profile = self.user_profiles[user_id]
        
        if np.sum(user_profile) == 0 or np.sum(movie_features) == 0:
            return 3.0
        
        similarity = np.dot(user_profile, movie_features)
        
        # Similarity'yi 1-5 rating aralığına dönüştür
        prediction = 1 + 4 * similarity  # 0-1 arası similarity'yi 1-5'e scale et
        
        return max(1, min(5, prediction))
    
    def predict_batch(self, test_pairs):
        """Toplu tahmin"""
        predictions = []
        print(f"🎭 {len(test_pairs)} content-based tahmin yapılıyor...")
        
        for user_id, movie_id in test_pairs:
            pred = self.predict(user_id, movie_id)
            predictions.append(pred)
        
        return np.array(predictions)

def evaluate_model(y_true, y_pred, model_name):
    """Model performansını değerlendir"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    print(f"\n📊 {model_name} Performansı:")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   MAE: {mae:.4f}")
    
    return {'rmse': rmse, 'mae': mae}

def cross_validation(model, train_df, user_item_matrix, movies_df=None, k_folds=5):
    """K-fold çapraz doğrulama"""
    print(f"\n🔄 {k_folds}-fold çapraz doğrulama başlıyor...")
    
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    rmse_scores = []
    mae_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_df)):
        print(f"   📁 Fold {fold + 1}/{k_folds}")
        
        # Bu fold için train ve validation setlerini al
        fold_train = train_df.iloc[train_idx]
        fold_val = train_df.iloc[val_idx]
        
        # Model türüne göre eğit
        if hasattr(model, 'user_similarity'):  # Collaborative Filtering
            # Bu fold için user-item matrix oluştur
            fold_matrix = fold_train.pivot_table(
                index='user_id', columns='movie_id', values='rating', fill_value=0
            )
            model.fit(fold_matrix)
        else:  # Content-Based Filtering
            model.fit(movies_df, fold_train)
        
        # Validation seti için tahmin yap
        val_pairs = [(row['user_id'], row['movie_id']) for _, row in fold_val.iterrows()]
        predictions = model.predict_batch(val_pairs)
        
        # Performansı değerlendir
        true_ratings = fold_val['rating'].values
        rmse = np.sqrt(mean_squared_error(true_ratings, predictions))
        mae = mean_absolute_error(true_ratings, predictions)
        
        rmse_scores.append(rmse)
        mae_scores.append(mae)
    
    print(f"\n✅ Çapraz doğrulama tamamlandı!")
    print(f"   Ortalama RMSE: {np.mean(rmse_scores):.4f} (±{np.std(rmse_scores):.4f})")
    print(f"   Ortalama MAE: {np.mean(mae_scores):.4f} (±{np.std(mae_scores):.4f})")
    
    return {
        'rmse_mean': np.mean(rmse_scores),
        'rmse_std': np.std(rmse_scores),
        'mae_mean': np.mean(mae_scores),
        'mae_std': np.std(mae_scores)
    }

def main():
    """Ana fonksiyon"""
    print("🎯 ÜÇÜNCÜ ADIM: MODEL GELİŞTİRME")
    print("="*50)
    
    # Veriyi yükle
    print("📥 Veriler yükleniyor...")
    user_item_matrix = pd.read_csv('data/processed/user_item_matrix.csv', index_col=0)
    train_df = pd.read_csv('data/processed/train_ratings.csv')
    test_df = pd.read_csv('data/processed/test_ratings.csv')
    movies_df = pd.read_csv('data/processed/movies.csv')
    
    print("✅ Veriler yüklendi!")
    
    # 1. Collaborative Filtering
    print(f"\n{'='*50}")
    print("🤝 COLLABORATIVE FILTERING")
    print("="*50)
    
    cf_model = CollaborativeFiltering(k_users=30)
    cf_model.fit(user_item_matrix)
    
    # Çapraz doğrulama
    cf_cv_results = cross_validation(cf_model, train_df, user_item_matrix, k_folds=3)
    
    # Test seti değerlendirmesi
    print("\n🧪 Test seti değerlendiriliyor...")
    test_pairs = [(row['user_id'], row['movie_id']) for _, row in test_df.iterrows()]
    cf_predictions = cf_model.predict_batch(test_pairs)
    cf_results = evaluate_model(test_df['rating'].values, cf_predictions, "Collaborative Filtering")
    
    # 2. Content-Based Filtering
    print(f"\n{'='*50}")
    print("🎬 CONTENT-BASED FILTERING")  
    print("="*50)
    
    cb_model = ContentBasedFiltering()
    cb_model.fit(movies_df, train_df)
    
    # Çapraz doğrulama
    cb_cv_results = cross_validation(cb_model, train_df, user_item_matrix, movies_df, k_folds=3)
    
    # Test seti değerlendirmesi
    cb_predictions = cb_model.predict_batch(test_pairs)
    cb_results = evaluate_model(test_df['rating'].values, cb_predictions, "Content-Based Filtering")
    
    # 3. Hibrit Model (Basit ağırlıklı ortalama)
    print(f"\n{'='*50}")
    print("🔄 HİBRİT MODEL")
    print("="*50)
    
    # %70 Collaborative + %30 Content-Based
    hybrid_predictions = 0.7 * cf_predictions + 0.3 * cb_predictions
    hybrid_results = evaluate_model(test_df['rating'].values, hybrid_predictions, "Hybrid Model")
    
    # 4. Sonuçları karşılaştır
    print(f"\n{'='*50}")
    print("📊 SONUÇLAR KARŞILAŞTIRMASI")
    print("="*50)
    
    results_df = pd.DataFrame({
        'Model': ['Collaborative Filtering', 'Content-Based', 'Hybrid'],
        'RMSE': [cf_results['rmse'], cb_results['rmse'], hybrid_results['rmse']],
        'MAE': [cf_results['mae'], cb_results['mae'], hybrid_results['mae']]
    })
    
    print(results_df.round(4))
    
    # En iyi modeli belirle
    best_model_idx = results_df['RMSE'].idxmin()
    best_model = results_df.loc[best_model_idx, 'Model']
    print(f"\n🏆 En iyi model: {best_model}")
    
    # 5. Modelleri kaydet
    print(f"\n💾 Modeller kaydediliyor...")
    with open('models/collaborative_model.pkl', 'wb') as f:
        pickle.dump(cf_model, f)
    
    with open('models/content_based_model.pkl', 'wb') as f:
        pickle.dump(cb_model, f)
    
    # Sonuçları kaydet
    results_df.to_csv('results/metrics/model_comparison.csv', index=False)
    
    print("✅ ÜÇÜNCÜ ADIM TAMAMLANDI!")
    print("🔜 Sonraki adım: Hiperparametre optimizasyonu")

if __name__ == "__main__":
    main()