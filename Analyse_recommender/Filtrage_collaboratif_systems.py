#!/usr/bin/env python
# coding: utf-8

# 			### SYSTEME DE RECOMMANDATIONS BASED SUR LE FILTRAGE COLLABORATIF ###

# * Objectif : Il s'agira de mettre en place un système de recommandations des produits complémentaires à un client lors de son achat ou de sa commande en ligne.
# 
# * Avantages: Ce système permettra à des sites e-commerce comme amazon, alibaba et autres d'augmenter leurs chiffres d'affaires et d'obtenir plus de revenu par clients et de comprendre le comportement des utilisateurs.
# 

# 						### Filtrage Collaboratif ###
# 
# * Le Filtrage Collaboratif est une méthode puissante pour les systèmes de recommandations, car il se base sur les interactions utilisateur-produit pour suggérer des produits similaires ou complémentaires ou pour prédire si un utilisateur pourrait aimer ce produit ou non.
# * Notons donc que cette technique est divisée en deux grandes approches :
# 	* Basé sur les utilisateurs(User-Based Collaborative Filtering)
# 	* Base sur les éléments (Item-based Collaborative Filtering)
# 
# * Pour cet notebook, nous irons étape par étape pour chaque approche et detaillé les principales étapes à suivre.

# * A - Importation des librairies 

import pandas as pd # pour le load data et manipuler les données
import numpy as np # pour manipuler les tableauet matrices
import seaborn as sns # pour la visualisation
import matplotlib.pyplot as plt 
import warnings, os, re, random
from sklearn.metrics.pairwise import cosine_similarity # pour evaluer la similarity 
from sklearn.metrics import precision_score, recall_score, confusion_matrix, f1_score


# * B - Chargement des données et affichages
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df.head(5)



# * C- Analyse descriptives et preparation des données

def analyse_descriptive(df):
    ## analyse générale des données
    print(df.describe())
    # les infos sur les données 
    print(f"Info data \n: {df.info()}")

    ## valeurs manquantes
    print(f'Nombres de valeurs manquantes : {df.isnull().sum()}')

    ## verifier les doublons dans les datas
    print(f"nombre de doublons :{df.duplicated().sum()}")


# * D- Analyse approfondies pour une meilleure compréhension

def analyse_data(df):
    ## top 20 des produits les plus commandées ou achetées par les clients

    top_produits = df.groupby('Description')['Quantity'].sum().reset_index().sort_values('Quantity', ascending = False)
    plt.figure(figsize = (10, 8))
    X = sns.barplot(x = 'Quantity', y = 'Description', data = top_produits.head(20))
    plt.title('Top 20 des produits most command')
    plt.tight_layout()
    plt.savefig('Top20_produits')
    plt.close()
    
    
    ## top 20 des clients avec plus de commandes
    top_customers = df.groupby('CustomerID')['Quantity'].sum().reset_index().sort_values('CustomerID', ascending=False)
    plt.figure(figsize=(12, 10))
    Y = sns.barplot(x = 'CustomerID', y = 'Quantity', hue='CustomerID', palette='viridis', data = top_customers.head(20))
    plt.title('Top20 des clients')
    plt.tight_layout()
    plt.savefig('Top20_clients')
    plt.close()
    for i in Y.containers:
        Y.bar_label(i)
  


    ## top des pays avec plus clients 

    top_customer_country = df.groupby('Country')['CustomerID'].count().reset_index().sort_values('CustomerID', ascending=False)
    plt.figure(figsize = (12, 10))
    Z= sns.barplot(x = 'CustomerID', y = 'Country', data = top_customer_country.head(20))
    plt.title('Top 20 des pays avec plus de clients')
    plt.tight_layout()
    plt.savefig('Top20_pays')
    plt.close()
    for x in Z.containers:
        Z.bar_label(x, label_type='center')
        
    return X, Y, Z 

def prix_total(df):
    ## calcul du prix total de ventes
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    return df.sample(5)

def plotting_analyse(df):
    ## pays avec plus de d'achats
    top_country_sales = df.groupby('Country')['TotalPrice'].sum().reset_index().sort_values('Country', ascending=False)
    plt.figure(figsize = (12, 10))
    A= sns.barplot(x = 'TotalPrice', y = 'Country', data = top_country_sales.head(20))
    plt.title('Top des pays avec plus de ventes')
    plt.tight_layout()
    plt.savefig('Top_ventes_pays')
    plt.close()
    for y in A.containers:
        A.bar_label(y, label_type='center')
        

    ## total price de produit vendu
    total_price = df.groupby('Description')['TotalPrice'].sum().reset_index().sort_values('TotalPrice', ascending=False)
    plt.figure(figsize=(12, 10))
    B = sns.barplot(x='TotalPrice', y='Description', data=total_price.head(20), hue='TotalPrice', palette='viridis')
    plt.title('Top 20 products sold')
    plt.xlabel('Total price')
    plt.ylabel('Product description')
    plt.tight_layout()
    plt.savefig('Top20_products_sales')
    plt.close()
    for y in B.containers:
        B.bar_label(y, label_type='edge')
        
    return A, B


# * Traitements des valeurs manquantes et feature enginéering

def treatment_data(df):


    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    ## fill the misisng valuees of description by most_frequent 
    most_frequent = df['Description'].mode()[0]
    df['Description'] = df['Description'].fillna(most_frequent)

    ## for customersID
    median_customer = df['CustomerID'].mean()
    df['CustomerID'] = df['CustomerID'].fillna(median_customer)

    ## extraction des years, date and month
    df['Year'] = df['InvoiceDate'].dt.year

    ## month
    df['Month'] = df['InvoiceDate'].dt.month
    #day
    df['Day'] = df['InvoiceDate'].dt.day

    df['CustomerID'] = df['CustomerID'].astype('int')
    
    return df



def plotting_after_treatment(df):
# * Quantité de produit vendu  par années, par mois et par jour

    top_year_sales = df.groupby('Year')['Quantity'].sum().reset_index()
    ## par mois
    top_month_sales = df.groupby('Month')['Quantity'].sum().reset_index()

    # par jour
    top_day_sales = df.groupby('Day')['Quantity'].sum().reset_index()

    fig, ax = plt.subplots(1, 3, figsize=(15, 6))
    sns.barplot(x='Year', y='Quantity', data=top_year_sales, ax=ax[0])
    sns.barplot(x='Month', y='Quantity', data=top_month_sales, ax=ax[1])
    sns.barplot(x = 'Day', y = 'Quantity', data = top_day_sales, ax=ax[2])
    ax[0].set_title('Yearly sales')
    ax[1].set_title(' Monthly sales')
    ax[2].set_title('Daily sales')
    plt.tight_layout()
    plt.savefig('Top20_produits_sales_month_year')
    plt.close()



    top_year_sales = df.groupby('Year')['Quantity'].mean().reset_index()
    ## par mois
    top_month_sales = df.groupby('Month')['Quantity'].mean().reset_index()

    # par jour
    top_day_sales = df.groupby('Day')['Quantity'].mean().reset_index()

    fig, ax = plt.subplots(1, 3, figsize=(15, 6))
    sns.barplot(x='Year', y='Quantity', data=top_year_sales, ax=ax[0])
    sns.barplot(x='Month', y='Quantity', data=top_month_sales, ax=ax[1])
    sns.barplot(x = 'Day', y = 'Quantity', data = top_day_sales, ax=ax[2])
    ax[0].set_title('Yearly sales moyenne')
    ax[1].set_title(' Monthly sales moyenne')
    ax[2].set_title('Daily sales moyenne')
    plt.tight_layout()
    plt.savefig('Topventes_year_month_daily')
    plt.close()

# * filtrez les produits dont les quantités sont < 0
def filtered_data(df):
    dff = df[df['Quantity'] >0]
    return dff


## CLients avec plus de chiffres d'affdaires

# * Nous allons utiliser les variables pertinentes telles que : InvoiceNo, description et Quantity pour le filtrage collaboratif afin d'evaluer l'experience client-produit
# ### Pour cela :
#  * realisons la matrice client-produit : chaque ligne represente un client et chaque colonne un produit ;
#  ** Le nombre de fois q'un produit a été acheté
#  ** un indicateur binaire (1: acheté, 0 : non acheté) 

def transform_data(dff):
    customer_product_matrix = dff.pivot_table(index='CustomerID', values='Quantity', columns="Description", aggfunc='sum').fillna(0)
    return customer_product_matrix


# * Evaluons la similarity client-client

def compute_similarity(customer_product_matrix):

    customer_similarity = cosine_similarity(customer_product_matrix)

    # convert in dataframe
    customer_similarity_df = pd.DataFrame(customer_similarity, index = customer_product_matrix.index, columns=customer_product_matrix.index)
    #customer_similarity_df.head(20)

    # * Evaluons la similarity produit-produit

    description_similarity = cosine_similarity(customer_product_matrix.T)
    description_similarity_df = pd.DataFrame(description_similarity, index= customer_product_matrix.columns, columns= customer_product_matrix.columns)
    #description_similarity_df.head(20)
    
    return customer_similarity_df, description_similarity_df

# ## Recommandations based sur les produits
# 
# * construis une fonction qui permet de fournir des recommandation basées sur les produits
# * decouvre les produits similaires à ceux commandés par le client
# * recommandez des produits avec des similarity élévées



def recommend_product(product_name, description_similarity_df, customer_product_matrix, num_recommends = 5):
    
    ## trouver les produits similaires
    similar_products = description_similarity_df[product_name].sort_values(ascending = False).index[1:]
    
    # obtenir les scores de popularité pour les produits similaires
    product_popularity = customer_product_matrix[similar_products].sum(axis=0)
    return product_popularity.sort_values(ascending = False).head(num_recommends)

#recommend_product(product_name="FELTCRAFT 6 FLOWER FRIENDS", description_similarity_df = description_similarity_df, customer_product_matrix=customer_product_matrix)


# ## Recommandations basé sur les Clients
# 
#  * Identifier les clients similaires
#  * Recommandez les produits achetés par les utilisateurs similaires mais pas encore l'utilisateur cible


def recommend_for_user(user_id, customer_similarity_df, customer_product_matrix, num_recommends = 5):
    ## identifier les clients similaires
    customer_similar = customer_similarity_df[user_id].sort_values(ascending = False).index[1:]
    ## popularity similiar
    customer_similar_popularity = customer_product_matrix[customer_similar].sum(axis = 0)
    
    # exclure les prodyuits achétes par la cible
    customer_products = customer_product_matrix.loc[user_id]
    recommendations = customer_similar_popularity[customer_products ==0].sort_values(ascending = False)
    return recommendations.head(num_recommends)

#recommend_for_user(user_id='17964', customer_similarity_df=customer_similarity_df, customer_product_matrix=customer_product_matrix)


# * Amelioration de la predsion des recommandations
# 
#  	* Pretraitment de la matrice customer-product
# 	* filtrez les produits ou clients peu frequents 

if __name__ == "__main__":
    file_path = '../data/online_retail.csv'
    z = load_data(file_path=file_path)
    analyse_descriptive(z)
    analyse_data(z)
    prix_total(z)
    plotting_analyse(z)
    dataset = treatment_data(z)
    
    plotting_after_treatment(dataset)
    data = filtered_data(dataset)
    customer_matrix_product = transform_data(data)
    customer_similarity_df, description_similarity_df = compute_similarity(customer_product_matrix=customer_matrix_product)
    #print(result)
    recommend_product(product_name="FELTCRAFT 6 FLOWER FRIENDS", description_similarity_df = description_similarity_df, customer_product_matrix=customer_matrix_product)
