# 📚 Book Recommendation System

> A data-driven recommendation engine that suggests books to users based on multiple filtering methodologies including collaborative filtering, content-based filtering, and weighted average scoring.

---

## 📌 Project Overview

This project builds a **Book Recommendation System** that leverages machine learning techniques to help users discover new books tailored to their preferences. The system supports both **guest** and **member** search experiences.

The content recommendation engine market was valued at **USD 2.18 Billion in 2020** and is projected to reach **USD 33.25 Billion by 2028**, growing at a CAGR of 40.58%. Studies show that 45% of consumers discover new books through online retailers' recommendation algorithms, highlighting the real-world impact of this technology.

---

## 📂 Dataset

The dataset was collected by **Cai-Nicolas Ziegler** over a period of 4 weeks and consists of 3 CSV files:

| File | Records | Features |
|------|---------|----------|
| `Users.csv` | 278,858 users | User ID, Location, Age |
| `Books.csv` | 271,379 books | ISBN, Title, Author, Year, Publisher, Image URLs |
| `Ratings.csv` | 1,149,780 ratings | User ID, ISBN, Book Rating |

**Source:** [Book-Crossing Dataset](http://www2.informatik.uni-freiburg.de/~cziegler/BX/)

---

## 🔧 Methodologies

### 1. Weighted Average
Calculates a score for each book using the formula:

```
Score = [votes × avg_rating / (votes + min_votes)] + [min_votes × mean_vote / (votes + min_votes)]
```

Used for general popularity-based recommendations, especially for guest users.

### 2. Item-Item Collaborative Filtering (KNN)
- Focuses on item-item relationships rather than user-user relationships
- Creates a matrix based on user ratings to identify similar items
- Number of neighbors considered: **5**

### 3. Content-Based Filtering

**SVD (Singular Value Decomposition)**
- Decomposes the ratings matrix into three matrices: **U**, **Σ**, and **V**
  - U → User factor matrix
  - Σ → Diagonal matrix containing singular values
  - V → Item factor matrix

**NMF (Non-negative Matrix Factorization)**
- Generates two non-negative matrices:
  - W → User feature matrix
  - H → Item feature matrix

### 4. Collaborative Filtering
- Based on the principle: *"people with similar preferences or behaviors can help predict a user's interests"*
- User-centric approach that does not require explicit knowledge about item characteristics
- Effective for handling **cold start problems**

---

## 🔄 Workflow

```
Raw Data (Users, Books, Ratings CSVs)
        ↓
  Data Preprocessing & Cleaning
        ↓
  Exploratory Data Analysis (EDA)
        ↓
  Model Building (KNN, SVD, NMF, Collaborative Filtering)
        ↓
  Evaluation (Hits@k, Recall@k)
        ↓
  Recommendation Output (Guest / Member Search)
```

---

## 📊 Evaluation Metrics

Models were evaluated using:
- **Hits@k count** — number of relevant items found in top-k recommendations
- **Recall@k** — proportion of relevant items retrieved in top-k results

These metrics were used to select the best matrix factorization method for the final system.

---

## 🖥️ Features

- **Guest Search** — Popularity-based recommendations using weighted average scoring
- **Member Search** — Personalized recommendations using collaborative and content-based filtering

---

## ⚠️ Limitations

- Performance may degrade for users with very sparse rating history
- Cold start problem for brand-new users with no prior interactions
- Dataset limited to a specific collection window (4 weeks)

---

## 🚀 Future Scope

- Incorporate deep learning models (e.g., Neural Collaborative Filtering)
- Add real-time recommendation updates as users interact with the system
- Expand the dataset with more recent book ratings
- Integrate natural language processing on book descriptions for richer content-based filtering

---

## 📚 References

1. T. Desai et al., "An enterprise-friendly book recommendation system for very sparse data," *CAST 2016*, doi: [10.1109/CAST.2016.7914968](https://doi.org/10.1109/CAST.2016.7914968)
2. M. F. Adak and M. Uçar, "A Book Recommendation System Using Decision Tree-based Fuzzy Logic for E-Commerce Sites," *HORA 2021*, doi: [10.1109/HORA52670.2021.9461319](https://doi.org/10.1109/HORA52670.2021.9461319)
3. P. Jomsri, "Book recommendation system for digital library based on user profiles by using association rule," *INTECH 2014*, doi: [10.1109/INTECH.2014.6927766](https://doi.org/10.1109/INTECH.2014.6927766)
4. L. Xin et al., "Collaborative Book Recommendation Based on Readers' Borrowing Records," *CBD 2013*, doi: [10.1109/CBD.2013.14](https://doi.org/10.1109/CBD.2013.14)
5. Y. Zhu, "A book recommendation algorithm based on collaborative filtering," *ICCSNT 2016*, doi: [10.1109/iccsnt.2016.8070165](https://doi.org/10.1109/iccsnt.2016.8070165)
6. Dataset: [Book-Crossing Dataset](http://www2.informatik.uni-freiburg.de/~cziegler/BX/)

---


