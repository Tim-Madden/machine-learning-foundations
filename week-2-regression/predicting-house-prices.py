import graphlab

sales = graphlab.SFrame('home_data.gl/')
graphlab.canvas.set_target('ipynb')

sales.show(view="Scatter Plot", x="sqft_living", y="price")

train_data,test_data = sales.random_split(.8,seed=0)

sqft_model = graphlab.linear_regression.create(train_data, target='price', features=['sqft_living'],validation_set=None)

print test_data['price'].mean()
print sqft_model.evaluate(test_data)


import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
plt.plot(test_data['sqft_living'],test_data['price'],'.',
        test_data['sqft_living'],sqft_model.predict(test_data),'-')


sqft_model.get('coefficients')

my_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'zipcode']
sales[my_features].show()

sales.show(view='BoxWhisker Plot', x='zipcode', y='price')

# #Build a regression model with more features
my_features_model = graphlab.linear_regression.create( train_data, target = 'price', features = my_features, validation_set = None)
print my_features
print sqft_model.evaluate(test_data)
print my_features_model.evaluate(test_data)

house1 = sales[sales['id']=='5309101200']
print house1['price']
print sqft_model.predict(house1)
print my_features_model.predict(house1)

# ##Last house, super fancy
bill_gates = {'bedrooms':[8], 
              'bathrooms':[25], 
              'sqft_living':[50000], 
              'sqft_lot':[225000],
              'floors':[4], 
              'zipcode':['98039'], 
              'condition':[10], 
              'grade':[10],
              'waterfront':[1],
              'view':[4],
              'sqft_above':[37500],
              'sqft_basement':[12500],
              'yr_built':[1994],
              'yr_renovated':[2010],
              'lat':[47.627606],
              'long':[-122.242054],
              'sqft_living15':[5000],
              'sqft_lot15':[40000]}

print my_features_model.predict(graphlab.SFrame(bill_gates))


# #Mean price in 98039 zipcode
sales_98039 = sales[sales['zipcode']=='98039']
print sales_98039['price'].mean()
# or in one line
print sales[sales['zipcode']=='98039']['price'].mean()


# #Logical Filter comparing number of rows
large_sales = sales[(sales['sqft_living']>2000) & (sales['sqft_living']<=4000)]
print large_sales.num_rows()/float(sales.num_rows())

# Training a model with more advanced features
advanced_features = [
'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'zipcode',
'condition', # condition of house				
'grade', # measure of quality of construction				
'waterfront', # waterfront property				
'view', # type of view				
'sqft_above', # square feet above ground				
'sqft_basement', # square feet in basement				
'yr_built', # the year built				
'yr_renovated', # the year renovated				
'lat', 'long', # the lat-long of the parcel				
'sqft_living15', # average sq.ft. of 15 nearest neighbors 				
'sqft_lot15', # average lot size of 15 nearest neighbors 
]

advanced_features_model = graphlab.linear_regression.create(train_data,target='price',features=advanced_features,validation_set=None)

my_RMSE = my_features_model.evaluate(test_data)['rmse']
advanced_RMSE = advanced_features_model.evaluate(test_data)['rmse']
print my_RMSE, advanced_RMSE, my_RMSE - advanced_RMSE





