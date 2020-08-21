# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 16:38:13 2020

@author: chetan
"""
import pandas as pd
from mlxtend.frequent_patterns import apriori,association_rules
import matplotlib.pyplot as plt
df = pd.read_csv("C:\\Users\\ADMIN\\Desktop\\chetan\\assignment\\7.association rules\\book.csv")

frequent_itemsets = apriori(df,min_support=0.005, max_len=3,use_colnames = True)
frequent_itemsets.shape

frequent_itemsets.sort_values('support',ascending = False,inplace=True)
plt.bar(x = list(range(1,5)),height = frequent_itemsets.support[1:5],color='rgmyk');plt.xticks(list(range(1,5)),frequent_itemsets.itemsets[1:5])
plt.xlabel('item-sets');plt.ylabel('support')

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules.shape


rules.head(10)
rules.sort_values('lift',ascending = False,inplace=True)

frequent_itemsets2 = apriori(df,min_support=0.001, max_len=2,use_colnames = True)
frequent_itemsets2.shape

plt.bar(x = list(range(1,5)),height = frequent_itemsets2.support[1:5],color='rgmyk');plt.xticks(list(range(1,5)),frequent_itemsets2.itemsets[1:5])
plt.xlabel('item-sets');plt.ylabel('support')

frequent_itemsets3 = apriori(df,min_support=0.01, max_len=3,use_colnames = True)
frequent_itemsets3.shape

plt.bar(x = list(range(1,5)),height = frequent_itemsets3.support[1:5],color='rgmyk');plt.xticks(list(range(1,5)),frequent_itemsets3.itemsets[1:5])
plt.xlabel('item-sets');plt.ylabel('support')