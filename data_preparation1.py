from main import *


#On va créer un seul dataset

jointure1 = pd.merge (commande_df, df_cmd_pdt, on="order_id")
jointure2 = pd.merge (jointure1, df_pdt, on="product_id")

#on a un Data set avec : le client (user_id) , le panier (order_id) , le product_id et la catégorie (department_id)
Data = jointure2
print(Data)

#On met le Data avec les variables qui nous interesse dans un csv
Data.to_csv("C:/Users/schatti/Bureau/Instacart Market Basket analyses/Data/data.csv")