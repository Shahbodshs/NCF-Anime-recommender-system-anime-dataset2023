## 📚 Neural Collaborative Filtering for Anime Recommendations

This project explores neural collaborative filtering (NCF) approaches for anime recommendation using the [MyAnimeList Dataset](https://www.kaggle.com/datasets/dbdmobile/myanimelist-dataset) from Kaggle.

### 🔬 Model Inspiration

The NCF architecture is inspired by the research paper:  
> **He, Xiangnan, et al. "Neural Collaborative Filtering."** *Proceedings of the 26th International Conference on World Wide Web*. [arXiv:1708.05031](https://arxiv.org/pdf/1708.05031)

This paper proposes a deep learning framework that unifies Generalized Matrix Factorization (GMF) and Multi-Layer Perceptron (MLP) under a single architecture. The idea is to capture both linear and non-linear user-item interactions.

---

### 🧠 Implemented Models
For implementation what we need to do is to both caputre the linear relationship between user and items and also capture the non linearity with Dense layers. 


#### 1. **NCF (GMF + MLP Unified Model)**

```python
def build_ncf_model(num_users, num_items, latent_dim=64, mlp_layers=[64, 32, 16]):
    # Input layers
    user_input = Input(shape=(1,), name='user_input')
    item_input = Input(shape=(1,), name='item_input')

    # GMF Branch
    gmf_user_embedding = Embedding(num_users, latent_dim, name='gmf_user_embedding')(user_input)
    gmf_item_embedding = Embedding(num_items, latent_dim, name='gmf_item_embedding')(item_input)
    gmf_output = Multiply()([Flatten()(gmf_user_embedding), Flatten()(gmf_item_embedding)])

    # MLP Branch
    mlp_user_embedding = Embedding(num_users, latent_dim, name='mlp_user_embedding')(user_input)
    mlp_item_embedding = Embedding(num_items, latent_dim, name='mlp_item_embedding')(item_input)
    mlp_concat = Concatenate()([Flatten()(mlp_user_embedding), Flatten()(mlp_item_embedding)])
    for layer_size in mlp_layers:
        mlp_concat = Dense(layer_size, activation='relu')(mlp_concat)

    # Combine branches
    merged = Concatenate()([gmf_output, mlp_concat])
    output = Dense(1, activation='sigmoid')(merged)

    return Model(inputs=[user_input, item_input], outputs=output)
```

Although the model is theoretically powerful, it underperformed in practice compared to a simpler baseline.

---

#### 2. **Baseline: Dot-Product RecommenderNet**

```python
def RecommenderNet(num_users, num_animes, embedding_size=128):
    user = Input(name='user_encoded', shape=[1])
    user_embedding = Embedding(name='user_embedding', input_dim=num_users, output_dim=embedding_size)(user)

    anime = Input(name='anime_encoded', shape=[1])
    anime_embedding = Embedding(name='anime_embedding', input_dim=num_animes, output_dim=embedding_size)(anime)

    dot_product = Dot(name='dot_product', normalize=True, axes=2)([user_embedding, anime_embedding])
    flattened = Flatten()(dot_product)
    dense = Dense(64, activation='relu')(flattened)
    output = Dense(1, activation='sigmoid')(dense)

    model = Model(inputs=[user, anime], outputs=output)
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=["mae", "mse"])
    return model
```

This model gave better results than the NCF implementation, likely due to its simplicity and effectiveness on the given dataset.

---

### 📊 Exploratory Data Analysis (EDA)

The `EDA.ipynb` notebook explores:

- 📈 **Genre popularity**
- 🎯 **Score distributions across anime types**
- 👥 **Gender distribution of users**
- 🎬 **Top studios by anime count**
- 📆 **Season & Year of anime releases**

Example EDA snippet:

```python
# Distribution of Anime Scores by Type
fig = px.box(df_anime, x='Type', y='Score', title='Distribution of Anime Scores by Type', color='Type')
fig.show()

# Distribution of Premiered Seasons
fig = go.Figure(data=go.Pie(labels=season_counts.index, values=season_counts.values, hole=0.4))
fig.update_layout(title='Distribution of Premiered Seasons')
fig.show()

# Top 10 Anime Studios by Production Volume
fig = go.Figure(data=go.Bar(x=top_studios.index, y=top_studios.values))
fig.update_layout(title='Number of Animes by Studio (Top 10)')
fig.show()
```

---

### 🚀 Potential Improvements

Some directions to enhance the performance and diversity of the recommender system:

- ✅ **Graph-Based Recommender Systems**  
  Use Graph Neural Networks (e.g., LightGCN, PinSAGE) to model complex relationships between users, items, genres, and studios.

- ✅ **Hybrid Recommendation**  
  Combine user-based collaborative filtering with neural models to capture both user similarity and deep representations.

- ✅ **Autoencoders / Variational Autoencoders**  
  Leverage unsupervised deep learning for embedding generation and dimensionality reduction.


---

📁 Feel free to check the code and EDA notebooks to explore more!
