from main import *
from data_preparation1 import *

def client_final(Data):
 

  L= []

  for client in Data.groupby ("user_id") :
        client_id = client [0]
        client_data = client[1]

        seq_client=[]

        for order in client_data.groupby("order_id"):
            order_id = order[0]
            order_data = order[1]
            departement_ids = list(order_data["department_id"])

            seq_client.append(departement_ids)

        L.append(seq_client)

  return L


 L= client_final(Data)