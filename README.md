# market-basket-analysis-for-ecommerce
Registration Numbers: 0410/2017, 0387/2017, 0388/2017, 0405/2017.

**Folders**
dataset: contains the dataset used.
ecm: contains our local ecommerce site.
marketbasketanaly.ipynb: our apriori algorithm file.

the project was to build an e commerce recommendation engine that uses apriori algorithm to establish associations between for frequently bought items. 
the algorithm will then recommend to customers products which were bought in the same cart by other customers depending on the support value of a given item in an itemsset.

**implementation of apriori algorithm using python**
In our project, we were to find the association between items bought on our ecommerce site and make recommendation of the next suitable item that could be bought if item A was bought.
We used a public dataset because our local ecommerce site had few transactions making it unsuitble for exploring associations of items in a given itemset.
other challenges we experienced were;
1. The system was hosted locally so we couldn't get more transactions
2. Our dataset was too little to use therefore causing the code not to run as expected hence having to use a public one


**Steps in performing market basket anaysis using apriori**
we install all the required libraries which include; apriori and mlxtend.
we load our dataset using pd.read_csv
we then view the top most part of our data using data.head function


**we use the following function to find all the unique values of our transactions reference columns in our dataframe**
Ref = data.Ref.unique()
print("AVAILABLE TRANSACTIONS")
print(Ref)


**grouping data columnwise data into groups of transaction reference and product description**
basket = (data.groupby(["Ref","Descr"])["Quantity"]).sum().unstack().reset_index().fillna(0).set_index('Ref')
basket


**defining the encoding functions to encode the split data**
def hot_encode(x):
    if(x<= 0):
        return 0
    if(x>= 1):
        return 1
        
        
**encoding the dataset using the encode functions and printing the encoded data**
basket_encoded = basket.applymap(hot_encode)
basket = basket_encoded
print("ENCODED")
print(basket)


**building the apriori model and setting the minimum support value to get frequent itemsets with their support values higher than the minimum support value**
frq_items = apriori(basket, min_support = 0.02, use_colnames = True)
print("Frequent Items")
print(frq_items)


**calculating the association rules using frequent itemsets and the data frame and support value to examine how strong the associations are**
rules = association_rules(frq_items, metric="lift", min_threshold=1)
rules.head()

**the final output**
From the association rules above, RED RETROSPOT CHARLOTTE BAG and  RED RETROSPORT CHARLOTTE BAG are items with the highest association with each other since these two items have the highest "lift" value. The higher the lift value, the higher the association between the items. If the lift value is more than 1, it is enough to say that those two items are associated with each other. In our case the highest value is 16.609974, which is very high. It means those two items are very good to be sold together. So, if a customer logs in the site and purchases a product, another product with the highest association with that particular product clicked or purchased product will be recommended to them.


