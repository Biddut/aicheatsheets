

```
import pandas as pd
```


```
import numpy as np


def make_minhash_signature(shingled_data, num_hashes):
  inv_index, docids = invert_shingles(shingled_data)
  num_docs = len(docids)
  
  # initialize the signature matrix with infinity in every entry
  sigmatrix = np.full([num_hash, num_docs], np.inf)
  
  # generate hash functions
  hash_funcs = make_hashes(num_hashes)
  
  # iterate over each non-zero entry of the characteristic matrix
  for row, docid in inv_index:
    # update signature matrix if needed 
    # THIS IS WHAT YOU NEED TO IMPLEMENT
  
  return sigmatrix, docids
```


      File "<ipython-input-4-a5c6b3e9d645>", line 19
        return sigmatrix, docids
             ^
    IndentationError: expected an indented block




```
def _make_vector_hash(num_hashes, m=4294967295):
    hash_fns = make_hashes(num_hashes)
    def _f(vec):
      acc = 0
      for i in range(len(vec)):
        h = hash_fns[i]
        acc += h(vec[i])
      return acc % m
    return _f
```


```
def minhash_similarity(id1, id2, minhash_sigmat, docids):
  # get column of the similarity matrix for the two documents
  # calculate the fraction of rows where two columns match
  # return this fraction as the minhash similarity estimate
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-7-5926457f3118> in <module>()
    ----> 1 _make_vector_hash(3)
    

    <ipython-input-6-3ff9d72c531e> in _make_vector_hash(num_hashes, m)
          1 def _make_vector_hash(num_hashes, m=4294967295):
    ----> 2     hash_fns = make_hashes(num_hashes)
          3     def _f(vec):
          4       acc = 0
          5       for i in range(len(vec)):


    NameError: name 'make_hashes' is not defined



```
%matplotlib inline
from matplotlib import pyplot as plt
import seaborn as sns # makes the graph prettier

s1 = "The cat sat on the mat."
s2 = "The red cat sat on the mat."

similarities = []
for shingle_size in range(2, 6):
    shingles1 = set([s1[max(0, i - shingle_size):i] for i in range(shingle_size, len(s1) + 1)])
    shingles2 = set([s2[max(0, i - shingle_size):i] for i in range(shingle_size, len(s2) + 1)])
    jaccard = len(shingles1 & shingles2) / len(shingles1 | shingles2)
    similarities.append(jaccard)

_ = plt.bar([2,3,4,5], similarities, width=0.25)
_ = plt.xlabel('Jaccard Similarity')
_ = plt.ylabel('Shingle Size')
```


![png](/.gitbook/assets/practice_Pad_146_0.png)



```
from collections import Counter

def count_runs(gList):
    """Count number of success runs of length k."""
    success_run = []
    count = 0
    for g in gList:
        if g == 1:
            count += 1
            #print(count)
        else:
            if count: success_run.append(count)
            count = 0
            #print(count)
   
    if count: success_run.append(count)
    #print(success_run)
    return Counter(success_run)
```


```
count_runs([0,1,0,1,1, 1])
```




    Counter({1: 1, 3: 1})




```
no_of_input=input()
numbers=input()
#print(no_of_input,numbers)
listNumber=[n for n in numbers.split(" ")]
print(listNumber)

def mean(listNumber):
  re
```

    2
    2 3
    ['2', '3']



```
import wordcloud
frequencies = { "Adam":2, "Brenda":3, "David":1, "Jose":3, "Charlotte":2, "Terry":1, "Robert":4}
cloud = wordcloud.WordCloud()
cloud.generate_from_frequencies(frequencies)
cloud.to_file("myfile.jpg")
```




    <wordcloud.wordcloud.WordCloud at 0x7f499aa62b00>




```
list = ['larry', 'curly', 'moe']
list.append('shemp')         ## append elem at end
list.insert(0, 'xxx')
list
```




    ['xxx', 'larry', 'curly', 'moe', 'shemp']




```
def combine_guests(guests1, guests2):
  new_dic={}
  new_dic.update(guests2)
  new_dic.update(guests1)
  return new_dic
 
  # Combine both dictionaries into one, with each key listed 
  # only once, and the value from guests1 taking precedence

Rorys_guests = { "Adam":2, "Brenda":3, "David":1, "Jose":3, "Charlotte":2, "Terry":1, "Robert":4}
Taylors_guests = { "David":4, "Nancy":1, "Robert":2, "Adam":1, "Samantha":3, "Chris":5}

print(combine_guests(Rorys_guests, Taylors_guests))
```

    {'David': 1, 'Nancy': 1, 'Robert': 4, 'Adam': 2, 'Samantha': 3, 'Chris': 5, 'Brenda': 3, 'Jose': 3, 'Charlotte': 2, 'Terry': 1}



```
tlist=[y for y in range(1,100) if y%3==0]
print(tlist)
```

    [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60, 63, 66, 69, 72, 75, 78, 81, 84, 87, 90, 93, 96, 99]



```
def format_address(address_string):

  house_number=""
  street_name=""
     
  address=address_string.split(" ")
  house_number=address[0]
 # print(address)
  
  # Declare variables
  
  # Separate the address string into parts

  # Traverse through the address parts
  #street_name=street_name.join(address,1:)
  for i in range(1,len(address)) :
    #print(address[i])
    street_name=street_name+address[i]
    street_name+=" "
   # print(street_name)
    #street_name=street_name.join(" ")
    # Determine if the address part is the
    # house number or part of the street name

  # Does anything else need to be done 
  # before returning the result?
  
  # Return the formatted string  
  return "house number {} on street named {}".format(house_number,street_name)

print(format_address("123 Main Street"))
# Should print: "house number 123 on street named Main Street"

print(format_address("1001 1st Ave"))
# Should print: "house number 1001 on street named 1st Ave"

print(format_address("55 North Center Drive"))
# Should print "house number 55 on street named North Center Drive"
```

    house number 123 on street named Main Street 
    house number 1001 on street named 1st Ave 
    house number 55 on street named North Center Drive 



```

```


```
wardrobe = {"shirt":["red","blue","white"], "jeans":["blue","black"]}
for wardrobeItems,itemcolorlist in wardrobe.items():
    #print(wardrobeItems,itemcolorlist)
	for color in itemcolorlist:
		print("{} {}".format(color,wardrobeItems))
```

    red shirt
    blue shirt
    white shirt
    blue jeans
    black jeans



```
language=["Java","Python","julia","c++"]
print(language)
length=[len(element) for element in language]
print(length)
```

    ['Java', 'Python', 'julia', 'c++']
    [4, 6, 5, 3]



```
a=4
print(a)

range(0,10)
print(range(0,10))
```

    4
    range(0, 10)



```
def skip_elements(elements):
	# Initialize variables
	new_list = []
	i = 0
  print(elements)
	# Iterate through the list
	for i in range(len(elements)):
    print(elements)
 # i=i+2
   
    
		# Does this element belong in the resulting list?
  	#	if  
		# Add this element to the resulting list
		#	___
		# Increment i
	
	#return i

print(skip_elements(["a", "b", "c", "d", "e", "f", "g"])) # Should be ['a', 'c', 'e', 'g']
print(skip_elements(['Orange', 'Pineapple', 'Strawberry', 'Kiwi', 'Peach'])) # Should be ['Orange', 'Strawberry', 'Peach']
print(skip_elements([])) # Should be []



```


      File "<ipython-input-24-3411aae6c716>", line 5
        print(elements)
                       ^
    IndentationError: unindent does not match any outer indentation level




```
elements2=["a", "b", "c", "d", "e", "f", "g"]

def skip_elements(elements):
  newList=[]
  i=0
  print(elements)
  for i in range(0,len(elements),i+2):
    newList+=(elements[i])
  print(newList)



skip_elements(elements2)
```

    ['a', 'b', 'c', 'd', 'e', 'f', 'g']
    ['a', 'c', 'e', 'g']



```
myPets = ['Zophie', 'Pooka', 'Fat-tail']
print('Enter a pet name:')
name = input()
if name not in myPets:
    print('I do not have a pet named ' + name)
else:
    print(name + ' is my pet.')
```

    Enter a pet name:
    sd
    I do not have a pet named sd



```
supplies = ['pens', 'staplers', 'flamethrowers', 'binders']
for  item in enumerate(supplies):
    print( item)
```

    (0, 'pens')
    (1, 'staplers')
    (2, 'flamethrowers')
    (3, 'binders')



```
import random
str="hello world"
#random.choice(supplies)
random.choice(str)
```




    'w'




```
import random
supplies = ['pens', 'staplers', 'flamethrowers', 'binders']
supplies[random.randint(0, len(supplies)-1)]
print (supplies.reverse())
supplies.sort
print (supplies)
```

    None
    ['binders', 'flamethrowers', 'staplers', 'pens']



```
def initials(phrase):
    words = phrase.split()
    result = ""
    for word in words:
        result += word[0]
    return result.upper()

print(initials("Universal Serial Bus")) # Should be: USB
print(initials("local area network")) # Should be: LAN
print(initials("Operating system")) # Should be: OS
```

    USB
    LAN
    OS



```

def is_palindrome(input_string):
	# We'll create two strings, to compare them
	new_string = ""
	reverse_string = ""
	# Traverse through each letter of the input string
	for a in input_string:
		new_string+=a.strip()
		reverse_string
		# Add any non-blank letters to the 
		# end of one string, and to the front
		# of the other string. 
		if ___:
			new_string = ___
			reverse_string = ___
	# Compare the strings
	if ___:
		return True
	return False

print(is_palindrome("Never Odd or Even")) # Should be True
print(is_palindrome("abc")) # Should be False
print(is_palindrome("kayak")) # Should be True
```


```
def is_palindrome(input_string):
	# We'll create two strings, to compare them
	new_string = ""
	reverse_string = ""
	# Traverse through each letter of the input string
	for char in len(input_string):
   
   reverse
		# Add any non-blank letters to the 
		# end of one string, and to the front
		# of the other string. 
		if ___:
			new_string = ___
			reverse_string = ___
	# Compare the strings
	if ___:
		return True
	return False

print(is_palindrome("Never Odd or Even")) # Should be True
print(is_palindrome("abc")) # Should be False
print(is_palindrome("kayak")) # Should be True
```


      File "<ipython-input-1-c92337487438>", line 6
        for ___:
               ^
    SyntaxError: invalid syntax




```
for x in range(1, 10, 3):
    print(x)
```

    1
    4
    7



```
for x in range(10):
    for y in range(x):
        print(y)
```


```
votes(["yes","no","maybe"])

def votes(params):
	for vote in params:
	    print("Possible option:" + vote)


```

    Possible option:yes
    Possible option:no
    Possible option:maybe



```
"big" > "small"
```




    False




```
11 % 5
```




    1




```
def sum(x, y):
		return(x+y)
print(sum(sum(1,2), sum(3,4)))
```

    10



```
((10 >= 5*2) and (10 <= 5*2))
```




    True




```
"big" > "small"
```




    False




```

list(range(0,10))
```




    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]




```
list(range(20,10,-2))
```




    [20, 18, 16, 14, 12]




```
l=list(range(0,10))
m=[]
for i in range(len(l)):
  m.append(l[i]+2)
else:
  print("print done")
print(m)
```

    print done
    [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]





```
import numpy as np

```


```
x=[1,2]
```


```
x=np.array(x)
```


```
x
```




    array([1, 2])



#SVD


```
# Singular-value decomposition
from numpy import array
from scipy.linalg import svd
# define a matrix
A = array([[1, 2], [3, 4], [5, 6], [7, 8]])
#print(A)
# SVD
U, s, VT = svd(A)
print(U.shape,s.shape,VT.shape)
print(U)
print(s)
print(VT)
```

    (4, 4) (2,) (2, 2)
    [[-0.15 -0.82 -0.39 -0.38]
     [-0.35 -0.42 0.24 0.80]
     [-0.55 -0.02 0.70 -0.46]
     [-0.74 0.38 -0.55 0.04]]
    [14.27 0.63]
    [[-0.64 -0.77]
     [0.77 -0.64]]



```
# Reconstruct SVD
from numpy import array
from numpy import diag
from numpy import dot
from numpy import zeros
from scipy.linalg import svd
# define a matrix
A = array([[1, 2], [3, 4], [5, 6],[7, 8]])
print(A)
# Singular-value decomposition
U, s, VT = svd(A)
print(U.shape,s.shape,VT.shape,A.shape)
# print(U)
# print(s)
print(diag(s))
print(VT)
print()
# create m x n Sigma matrix
Sigma = zeros((A.shape[0], A.shape[1]))
print(Sigma.shape)
# populate Sigma with n x n diagonal matrix
Sigma[:A.shape[1], :A.shape[1]] = diag(s)
print(Sigma.shape)
# reconstruct matrix
B = U.dot(Sigma.dot(VT))
print(B)
```

    [[1 2]
     [3 4]
     [5 6]
     [7 8]]
    (4, 4) (2,) (2, 2) (4, 2)
    [[14.27 0.00]
     [0.00 0.63]]
    [[-0.64 -0.77]
     [0.77 -0.64]]
    
    (4, 2)
    (4, 2)
    [[1.00 2.00]
     [3.00 4.00]
     [5.00 6.00]
     [7.00 8.00]]



```
# populate Sigma with n x n diagonal matrix
Sigma[:A.shape[1]].shape
```




    (2, 2)




```

```




    array([3, 4])




```
A.shape[0]
```




    4




```
from numpy import array
from numpy import diag
from numpy import zeros
from scipy.linalg import svd
# define a matrix
A = array([
	[1,2,3,4,5,6,7,8,9,10],
	[11,12,13,14,15,16,17,18,19,20],
	[21,22,23,24,25,26,27,28,29,30]])
#print(A)
# Singular-value decomposition
U, s, VT = svd(A)

print(U.shape, s.shape, VT.shape,A.shape )

# create m x n Sigma matrix
Sigma = zeros((A.shape[0], A.shape[1]))

# populate Sigma with n x n diagonal matrix
Sigma[:A.shape[0], :A.shape[0]] = diag(s)
#print(Sigma)
# select
n_elements = 2

Sigma = Sigma[:, :n_elements]
VT = VT[:n_elements, :]

# reconstruct
B = U.dot(Sigma.dot(VT))
print(U.shape, s.shape, VT.shape,A.shape )
print(B)
# transform
T = U.dot(Sigma)
print(T)
T = A.dot(VT.T)
print(T)
```

    (3, 3) (3,) (10, 10) (3, 10)
    (3, 3) (3,) (2, 10) (3, 10)
    [[1.00 2.00 3.00 4.00 5.00 6.00 7.00 8.00 9.00 10.00]
     [11.00 12.00 13.00 14.00 15.00 16.00 17.00 18.00 19.00 20.00]
     [21.00 22.00 23.00 24.00 25.00 26.00 27.00 28.00 29.00 30.00]]
    [[-18.52 6.48]
     [-49.81 1.91]
     [-81.10 -2.65]]
    [[-18.52 6.48]
     [-49.81 1.91]
     [-81.10 -2.65]]



```
from numpy import array
from numpy import diag
from numpy import zeros
from scipy.linalg import svd
```


```
dff = pd.read_csv("iris.data", names=['sepal length','sepal width','petal length','petal width','target'])
```


```
dff.shape
```




    (150, 5)




```
features = ['sepal length', 'sepal width', 'petal length', 'petal width']
x = dff.loc[:, features]
```


```
U, s, VT = svd(x)
```


```
U.shape,s.shape,VT.shape,x.shape
```




    ((150, 150), (4,), (4, 4), (150, 4))




```
s.shape

```




    (4,)




```
# create m x n Sigma matrix
Sigma = zeros((x.shape[0], x.shape[0]))
Sigma.shape

```




    (150, 150)




```
# populate Sigma with n x n diagonal matrix
Sigma[:x.shape[1], :x.shape[1]]=diag(s)
```


```
# select
n_elements = 2

Sigma = Sigma[:, :n_elements]
VT = VT[:n_elements, :]
```


```
Sigma.shape
```




    (150, 2)




```
VT
```




    array([[-0.75, -0.38, -0.51, -0.17],
           [0.29, 0.54, -0.71, -0.34]])




```
# reconstruct
B = U.dot(Sigma.dot(VT))
print(B)
```


```
# transform
T = U.dot(Sigma)
print(T)

T = x.dot(VT.T)
print(T)
```


```
VT.T.shape
```




    (4, 2)




```

```

                0         1
    0   -5.912204  2.303442
    1   -5.572076  1.973831
    2   -5.446485  2.096533
    3   -5.436019  1.871681
    4   -5.875066  2.329348
    ..        ...       ...
    145 -9.226698 -0.929481
    146 -8.566555 -1.036575
    147 -9.026101 -0.883220
    148 -9.105660 -0.996221
    149 -8.490509 -0.914877
    
    [150 rows x 2 columns]



```
import numpy as np
from numpy.linalg import svd

# define your matrix as a 2D numpy array
A = np.array([[4, 0,3,5], [3, -5,6,5], [35, -5,4,3], [3, -7,6,8], [3, -7,6,8], [3, -7,6,8], [3, -7,6,8], [3, -7,6,8], [3, -7,6,8], [3, -7,6,8]])

U, S, VT = svd(A)

print("Left Singular Vectors:")
print(A.shape,U.shape)
print("Singular Values:") 
print(np.diag(S))
print("Right Singular Vectors:") 
print(VT)

# check that this is an exact decomposition
# @ is used for matrix multiplication in Py3, use np.matmul with Py2
#print(U @ np.diag(S) @ VT)
```

    Left Singular Vectors:
    (10, 4) (10, 10)
    Singular Values:
    [[42.42 0.00 0.00 0.00]
     [0.00 26.67 0.00 0.00]
     [0.00 0.00 3.48 0.00]
     [0.00 0.00 0.00 1.72]]
    Right Singular Vectors:
    [[-0.74 0.39 -0.35 -0.41]
     [-0.67 -0.39 0.36 0.52]
     [-0.04 -0.81 -0.13 -0.57]
     [-0.03 0.20 0.86 -0.48]]



```
Sigma=np.diag(S)
```


```
Sigma
```




    array([[42.41963146,  0.        ],
           [ 0.        , 26.67440613],
           [ 0.        ,  0.        ],
           [ 0.        ,  0.        ]])




```
n_elements = 2

Sigma = Sigma[:, :n_elements]
VT = VT[:n_elements, :]
```


```
VT
```




    array([[-0.74475647,  0.39078191, -0.3482    , -0.41398559],
           [-0.66526175, -0.39356261,  0.3604708 ,  0.52210734]])




```
# reconstruct
B = U.dot(Sigma.dot(VT))
print(B.T)
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-114-a413cd81f7d5> in <module>()
          1 # reconstruct
    ----> 2 B = U.dot(Sigma.dot(VT))
          3 print(B.T)


    ValueError: shapes (10,10) and (4,4) not aligned: 10 (dim 1) != 4 (dim 0)



```
import numpy as np
from sklearn.datasets import load_digits
from matplotlib import pyplot as plt
from sklearn.decomposition import TruncatedSVD
float_formatter = lambda x: "%.2f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})
from sklearn.ensemble import RandomForestClassifier
```


```
X, y = load_digits(return_X_y=True)
```


```
X[0]
```




    array([0.00, 0.00, 5.00, 13.00, 9.00, 1.00, 0.00, 0.00, 0.00, 0.00, 13.00,
           15.00, 10.00, 15.00, 5.00, 0.00, 0.00, 3.00, 15.00, 2.00, 0.00,
           11.00, 8.00, 0.00, 0.00, 4.00, 12.00, 0.00, 0.00, 8.00, 8.00, 0.00,
           0.00, 5.00, 8.00, 0.00, 0.00, 9.00, 8.00, 0.00, 0.00, 4.00, 11.00,
           0.00, 1.00, 12.00, 7.00, 0.00, 0.00, 2.00, 14.00, 5.00, 10.00,
           12.00, 0.00, 0.00, 0.00, 0.00, 6.00, 13.00, 10.00, 0.00, 0.00,
           0.00])




```
image = X[1]
```


```
image = image.reshape((8, 8))
plt.matshow(image, cmap = 'gray')
```




    <matplotlib.image.AxesImage at 0x7f7f4be7de80>




![png](assets/practice_Pad_75_1.png)



```
image
```




    array([[0.00, 0.00, 0.00, 12.00, 13.00, 5.00, 0.00, 0.00],
           [0.00, 0.00, 0.00, 11.00, 16.00, 9.00, 0.00, 0.00],
           [0.00, 0.00, 3.00, 15.00, 16.00, 6.00, 0.00, 0.00],
           [0.00, 7.00, 15.00, 16.00, 16.00, 2.00, 0.00, 0.00],
           [0.00, 0.00, 1.00, 16.00, 16.00, 3.00, 0.00, 0.00],
           [0.00, 0.00, 1.00, 16.00, 16.00, 6.00, 0.00, 0.00],
           [0.00, 0.00, 1.00, 16.00, 16.00, 6.00, 0.00, 0.00],
           [0.00, 0.00, 0.00, 11.00, 16.00, 10.00, 0.00, 0.00]])




```
U, s, V = np.linalg.svd(image, full_matrices=False)
S = np.zeros((image.shape[0], image.shape[1]))
S[:image.shape[0], :image.shape[0]] = np.diag(s)
print(U.shape,s.shape,V.shape)
n_component = 2
S = S[:, :n_component]
VT = VT[:n_component, :]
A = U.dot(S.dot(VT))

print(A)
```

    (8, 8) (8,) (8, 8)
    [[-12.02 8.03 -7.18 -8.75]
     [-12.60 9.78 -8.76 -10.81]
     [-16.85 9.08 -8.09 -9.65]
     [-27.42 4.03 -3.45 -2.80]
     [-16.29 9.08 -8.10 -9.70]
     [-16.10 9.84 -8.79 -10.62]
     [-16.10 9.84 -8.79 -10.62]
     [-12.54 10.04 -8.99 -11.12]]


#LDA


```
# test classification dataset
from sklearn.datasets import make_classification
# define dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=10, n_redundant=0, random_state=1)
# summarize the dataset
print(X.shape, y.shape)
```

    (1000, 10) (1000,)



```
y

```


```

# evaluate a lda model on the dataset
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# define dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=10, n_redundant=0, random_state=1)
# define model
model = LinearDiscriminantAnalysis()
# define model evaluation method
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# summarize result
print('Mean Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
# evaluate a lda model on the dataset
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# define dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=10, n_redundant=0, random_state=1)
# define model
model = LinearDiscriminantAnalysis()
# define model evaluation method
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# summarize result
print('Mean Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
```

    Mean Accuracy: 0.893 (0.033)
    Mean Accuracy: 0.893 (0.033)



```
# grid search solver for lda
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# define dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=10, n_redundant=0, random_state=1)
# define model
model = LinearDiscriminantAnalysis()
# define model evaluation method
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# define grid
grid = dict()
grid['solver'] = ['svd', 'lsqr', 'eigen']
# define search
search = GridSearchCV(model, grid, scoring='accuracy', cv=cv, n_jobs=-1)
# perform the search
results = search.fit(X, y)
# summarize
print('Mean Accuracy: %.3f' % results.best_score_)
print('Config: %s' % results.best_params_)
```

    Mean Accuracy: 0.893
    Config: {'solver': 'svd'}



```
class LDA:
    def __init__(self):
        pass

    def fit(self, X, y):
        target_classes = np.unique(y)
        print(target_classes)
        mean_vectors = []
 
        for cls in target_classes:
            mean_vectors.append(np.mean(X[y == cls], axis=0))
        
        if len(target_classes) < 3:
            mu1_mu2 = (mean_vectors[0] - mean_vectors[1]).reshape(1, X.shape[1])
            B = np.dot(mu1_mu2.T, mu1_mu2)
        else:
            data_mean = np.mean(X, axis=0).reshape(1, X.shape[1])
            B = np.zeros((X.shape[1], X.shape[1]))
            for i, mean_vec in enumerate(mean_vectors):
                n = X[y == i].shape[0]
                mean_vec = mean_vec.reshape(1, X.shape[1])
                mu1_mu2 = mean_vec - data_mean

                B += n * np.dot(mu1_mu2.T, mu1_mu2)
        
        s_matrix=[]
        for cls, mean in enumerate(mean_vectors):
          Si = np.zeros((X.shape[1], X.shape[1]))
          for row in X[y == cls]:
              t = (row - mean).reshape(1, X.shape[1])
              Si += np.dot(t.T, t)
          s_matrix.append(Si)
        
        S = np.zeros((X.shape[1], X.shape[1]))
        for s_i in s_matrix:
            S += s_i
        
        S_inv = np.linalg.inv(S)
        S_inv_B = S_inv.dot(B)

        eig_vals, eig_vecs = np.linalg.eig(S_inv_B)

        idx = eig_vals.argsort()[::-1]
        eig_vals = eig_vals[idx]
        eig_vecs = eig_vecs[:, idx]

        return eig_vecs
```


```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
import seaborn as sns


def load_data(cols, load_all=False, head=False):
    iris = sns.load_dataset("iris")

    if not load_all:
        if head:
            iris = iris.head(100)
        else:
            iris = iris.tail(100)

    le = preprocessing.LabelEncoder()
    y = le.fit_transform(iris["species"])

    X = iris.drop(["species"], axis=1)

    if len(cols) > 0:
        X = X[cols]

    return X.values, y

# Experiment 1
cols = ["petal_length", "petal_width"]
X, y = load_data(cols, load_all=False, head=True)
print(X.shape,type(X),type(y))
print(y)
lda = LDA()
eig_vecs = lda.fit(X, y)
W = eig_vecs[:, :1]

print(W)
colors = ['red', 'green', 'blue']
fig, ax = plt.subplots(figsize=(10, 8))

for point, pred in zip(X, y):
    ax.scatter(point[0], point[1], color=colors[pred], alpha=0.3)
    proj = (np.dot(point, W) * W) / np.dot(W.T, W)

    ax.scatter(proj[0], proj[1], color=colors[pred], alpha=0.3)
 
plt.show()

```


```
df = pd.read_csv("iris.data", names=['sepal length','sepal width','petal length','petal width','target'])
```


```
df.head(100)
```


```
from sklearn import preprocessing
```


```
features = ['sepal length', 'sepal width', 'petal length', 'petal width']
features2=["petal length", "petal width",'sepal length']
X = df.loc[:, features]
r = df.loc[:, 'target']
```


```
le = preprocessing.LabelEncoder()
y = le.fit_transform(r)
```


```
y
```


```
X=np.array(X)
```


```
lda = LDA()
eig_vecs = lda.fit(X, y)
W = eig_vecs[:, :2]

print(W)
```

    [0 1 2]
    [[ 0.20490976 -0.00898234]
     [ 0.38714331 -0.58899857]
     [-0.54648218  0.25428655]
     [-0.71378517 -0.76703217]]



```
transformed=X.dot(W)
```


```
transformed
```


```
plt.scatter(transformed[:, 0], transformed[:, 1], c=y, cmap=plt.cm.Set1)
plt.show()
```


![png](assets/practice_Pad_95_0.png)



```
projected_mat=[]
for point in X:
  proj = (np.dot(point, W) * W) / np.dot(W.T, W)
  #print(type(proj))
  projected_mat.append(proj)

```


```
projected_mat
```


```
proj
```




    array([[2.90993767],
           [2.621169  ]])




```
colors = ['red', 'green', 'blue']
fig, ax = plt.subplots(figsize=(10, 8))
for point,proj, pred in zip(X,projected_mat,y):
  ax.scatter(point[0], point[1], color=colors[pred], alpha=0.5)
  ax.scatter(proj[0], proj[1], color=colors[pred], alpha=0.3)
```


![png](assets/practice_Pad_99_0.png)



```
colors = ['red', 'green', 'blue']
fig, ax = plt.subplots(figsize=(10, 8))

for point, pred in zip(X, y):
    ax.scatter(point[0], point[1], color=colors[pred], alpha=0.5)
    #proj = (np.dot(point, W) * W) / np.dot(W.T, W)

    #ax.scatter(proj[0], proj[1], color=colors[pred], alpha=0.3)
 
plt.show()
```

#clustering


```
import sys

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
import numpy as np
import sklearn

from sklearn.datasets.samples_generator import make_blobs
from sklearn.metrics.pairwise import euclidean_distances

# clustering libraries
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans

# alternative scipy implementation for clustering
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram

# for scaling numpy array
from sklearn.preprocessing import StandardScaler

givenDec = lambda gdVal: float('%.1f' % gdVal) # 1 digit
```

    /usr/local/lib/python3.6/dist-packages/sklearn/utils/deprecation.py:144: FutureWarning: The sklearn.datasets.samples_generator module is  deprecated in version 0.22 and will be removed in version 0.24. The corresponding classes / functions should instead be imported from sklearn.datasets. Anything that cannot be imported from sklearn.datasets is now part of the private API.
      warnings.warn(message, FutureWarning)



```
x2d, y2d_true = make_blobs(n_samples=100, centers=3, cluster_std=0.50, random_state=0)
c1 = [-1.0,3.0]
c2 = [1.0,5.0]
c3 = [2.0,1.5]
cc = [c1,c2,c3]
```


```
cc=np.array(cc)
```


```
print(type(x2d), type(y2d_true),type(cc))
```

    <class 'numpy.ndarray'> <class 'numpy.ndarray'> <class 'numpy.ndarray'>



```
plt.scatter(x2d[:, 0], x2d[:, 1], s=50) # s=50 is size of the blobs
plt.scatter(np.array(cc)[:, 0], np.array(cc)[:, 1], s=100) # plot cluster centers
```


```
import math
# assume data is 2D
def getClusterCentre(gvCenters, gvPt):
    min_distance = 100000
    i_centre = -1
    for i,cc in enumerate(gvCenters):
        print(i,cc)
        #tmp_dist = math.sqrt(pow(cc[0] - gvPt[0], 2) + pow(cc[1] - gvPt[1], 2))
        tmp_dist = consine_distance(cc,gvPt)
        if(tmp_dist < min_distance):
            min_distance = tmp_dist
            i_centre = i
    return i_centre
```


```
n_iter = 3

#print("cc", cc)
for n in range(n_iter):
    print("iter", n)
    #y2d_predict = {}
    clusters = {k: [] for k in range(len(cc))}

    for i,x in enumerate(x2d):
        clusters[getClusterCentre(cc, x)].append(list(x))
    #print(len(clusters[0]))
    

    # updated cluster centres:
    new_cc = [None]*len(cc)
    for i in range(len(new_cc)):
        #print(i,clusters[i],'\n\n')
        new_cc[i] = list(np.around(np.sum(clusters[i],axis=0)/len(clusters[i]),3))

    print("new_cc", new_cc)
    # check convergence
    diff = sum(sum(abs(np.array(cc) - np.array(new_cc))))
    if diff > 0:
        cc = new_cc
    else:
        print("CONVERGED!")
        break
    #print("cc",cc)

print("cc",cc)
```


```
 new_cc[0] =list(np.around(np.sum(clusters[0],axis=0)/len(clusters[0]),3))
 new_cc[0]
```


```
 new_cc[0] =list(np.around(np.sum(clusters[0],axis=0)/len(clusters[0]),3))
```


```
new_cc
```




    []




```
len(clusters[0])
```




    25




```
def consine_distance(a,b):
  d_tmp=np.dot(a,b)
  a_tmp=math.sqrt(sum(np.square(a)))
  b_tmp=math.sqrt(sum(np.square(b)))
  d=1-(d_tmp/(a_tmp*b_tmp))
  #print(d_tmp,a_tmp,a_tmp)
  #print(d)
  return d

```


```
# problem data
import numpy as np
from sklearn.datasets.samples_generator import make_blobs

n_clusters = 4
x4d, y4d_true = make_blobs(n_samples=100, n_features = 4, centers=n_clusters, cluster_std=2.50, random_state=0)

# round data points to one decimal
x4d = np.round(np.array(x4d),1)

orig_cc = [None]*4

# get original cluster centres
clusters = {k: [] for k in range(len(orig_cc))} # initialize clusters
for i,x in enumerate(x4d):
    clusters[y4d_true[i]].append(list(x))

for i in range(len(orig_cc)):
    orig_cc[i] = list(np.around(np.mean(clusters[i], axis=0),2))

print("orig_cc", np.round(orig_cc,3))

# initial cluster centres
c1 = [-2.6, 7.5, 0.4, 8.9] # x4d[1]
c2 = [4.2, 2.8, 5.6, 3.1] # x4d[2]
c3 = [8.9, -3.1, 5.1, -3.6] # x4d[3]
c4 = [-2.2, 4.9, 1.1, 7.4] # x4d[4]
```

    orig_cc [[ 0.66  4.5   2.02  1.38]
     [-1.72  3.56 -1.02  7.37]
     [ 8.54 -1.88  5.52  0.15]
     [ 0.72  8.4  -8.92 -9.15]]



```
x4d
```


```
y4d_true
```


```
cc=[c1,c2,c3,c4]
```


```
cc=np.array(cc)
```


```
new_cc
```


```
n_iter = 30

#print("cc", cc)
for n in range(n_iter):
    print("iter", n)
    #y2d_predict = {}
    clusters = {k: [] for k in range(len(cc))}

    for i,x in enumerate(x4d):
      print(getClusterCentre(cc, x))
      #clusters[getClusterCentre(cc, x)].append(list(x))
    #print(len(clusters[0]))
    

    # updated cluster centres:
    new_cc = [None]*len(cc)
    for i in range(len(new_cc)):
        #print(i,clusters[i],'\n\n')
        #new_cc[i] = list(np.around(np.sum(clusters[i],axis=0)/len(clusters[i]),3))
        new_cc[i] = list(np.around(np.mean(clusters[i],axis=0),3))

    print("new_cc", new_cc)
    # check convergence
    diff = sum(sum(abs(np.array(cc) - np.array(new_cc))))
    if diff > 0:
        cc = new_cc
    else:
        print("CONVERGED!")
        break
    #print("cc",cc)

print("cc",cc)
```

    iter 0
    0 [-2.6  7.5  0.4  8.9]
    1 [4.2 2.8 5.6 3.1]
    2 [ 8.9 -3.1  5.1 -3.6]
    3 [-2.2  4.9  1.1  7.4]
    2
    0 [-2.6  7.5  0.4  8.9]
    1 [4.2 2.8 5.6 3.1]
    2 [ 8.9 -3.1  5.1 -3.6]
    3 [-2.2  4.9  1.1  7.4]
    0
    0 [-2.6  7.5  0.4  8.9]
    1 [4.2 2.8 5.6 3.1]
    2 [ 8.9 -3.1  5.1 -3.6]
    3 [-2.2  4.9  1.1  7.4]
    1
    0 [-2.6  7.5  0.4  8.9]
    1 [4.2 2.8 5.6 3.1]
    2 [ 8.9 -3.1  5.1 -3.6]
    3 [-2.2  4.9  1.1  7.4]
    2
    0 [-2.6  7.5  0.4  8.9]
    1 [4.2 2.8 5.6 3.1]
    2 [ 8.9 -3.1  5.1 -3.6]
    3 [-2.2  4.9  1.1  7.4]
    3
    0 [-2.6  7.5  0.4  8.9]
    1 [4.2 2.8 5.6 3.1]
    2 [ 8.9 -3.1  5.1 -3.6]
    3 [-2.2  4.9  1.1  7.4]
    1
    0 [-2.6  7.5  0.4  8.9]
    1 [4.2 2.8 5.6 3.1]
    2 [ 8.9 -3.1  5.1 -3.6]
    3 [-2.2  4.9  1.1  7.4]
    3
    0 [-2.6  7.5  0.4  8.9]
    1 [4.2 2.8 5.6 3.1]
    2 [ 8.9 -3.1  5.1 -3.6]
    3 [-2.2  4.9  1.1  7.4]
    0
    0 [-2.6  7.5  0.4  8.9]
    1 [4.2 2.8 5.6 3.1]
    2 [ 8.9 -3.1  5.1 -3.6]
    3 [-2.2  4.9  1.1  7.4]
    2
    0 [-2.6  7.5  0.4  8.9]
    1 [4.2 2.8 5.6 3.1]
    2 [ 8.9 -3.1  5.1 -3.6]
    3 [-2.2  4.9  1.1  7.4]
    2
    0 [-2.6  7.5  0.4  8.9]
    1 [4.2 2.8 5.6 3.1]
    2 [ 8.9 -3.1  5.1 -3.6]
    3 [-2.2  4.9  1.1  7.4]
    0
    0 [-2.6  7.5  0.4  8.9]
    1 [4.2 2.8 5.6 3.1]
    2 [ 8.9 -3.1  5.1 -3.6]
    3 [-2.2  4.9  1.1  7.4]
    2
    0 [-2.6  7.5  0.4  8.9]
    1 [4.2 2.8 5.6 3.1]
    2 [ 8.9 -3.1  5.1 -3.6]
    3 [-2.2  4.9  1.1  7.4]
    2
    0 [-2.6  7.5  0.4  8.9]
    1 [4.2 2.8 5.6 3.1]
    2 [ 8.9 -3.1  5.1 -3.6]
    3 [-2.2  4.9  1.1  7.4]
    2
    0 [-2.6  7.5  0.4  8.9]
    1 [4.2 2.8 5.6 3.1]
    2 [ 8.9 -3.1  5.1 -3.6]
    3 [-2.2  4.9  1.1  7.4]
    0
    0 [-2.6  7.5  0.4  8.9]
    1 [4.2 2.8 5.6 3.1]
    2 [ 8.9 -3.1  5.1 -3.6]
    3 [-2.2  4.9  1.1  7.4]
    2
    0 [-2.6  7.5  0.4  8.9]
    1 [4.2 2.8 5.6 3.1]
    2 [ 8.9 -3.1  5.1 -3.6]
    3 [-2.2  4.9  1.1  7.4]
    0
    0 [-2.6  7.5  0.4  8.9]
    1 [4.2 2.8 5.6 3.1]
    2 [ 8.9 -3.1  5.1 -3.6]
    3 [-2.2  4.9  1.1  7.4]
    3
    0 [-2.6  7.5  0.4  8.9]
    1 [4.2 2.8 5.6 3.1]
    2 [ 8.9 -3.1  5.1 -3.6]
    3 [-2.2  4.9  1.1  7.4]
    0
    0 [-2.6  7.5  0.4  8.9]
    1 [4.2 2.8 5.6 3.1]
    2 [ 8.9 -3.1  5.1 -3.6]
    3 [-2.2  4.9  1.1  7.4]
    0
    0 [-2.6  7.5  0.4  8.9]
    1 [4.2 2.8 5.6 3.1]
    2 [ 8.9 -3.1  5.1 -3.6]
    3 [-2.2  4.9  1.1  7.4]
    0
    0 [-2.6  7.5  0.4  8.9]
    1 [4.2 2.8 5.6 3.1]
    2 [ 8.9 -3.1  5.1 -3.6]
    3 [-2.2  4.9  1.1  7.4]
    1
    0 [-2.6  7.5  0.4  8.9]
    1 [4.2 2.8 5.6 3.1]
    2 [ 8.9 -3.1  5.1 -3.6]
    3 [-2.2  4.9  1.1  7.4]
    2
    0 [-2.6  7.5  0.4  8.9]
    1 [4.2 2.8 5.6 3.1]
    2 [ 8.9 -3.1  5.1 -3.6]
    3 [-2.2  4.9  1.1  7.4]
    2
    0 [-2.6  7.5  0.4  8.9]
    1 [4.2 2.8 5.6 3.1]
    2 [ 8.9 -3.1  5.1 -3.6]
    3 [-2.2  4.9  1.1  7.4]
    3
    0 [-2.6  7.5  0.4  8.9]
    1 [4.2 2.8 5.6 3.1]
    2 [ 8.9 -3.1  5.1 -3.6]
    3 [-2.2  4.9  1.1  7.4]
    2
    0 [-2.6  7.5  0.4  8.9]
    1 [4.2 2.8 5.6 3.1]
    2 [ 8.9 -3.1  5.1 -3.6]
    3 [-2.2  4.9  1.1  7.4]
    2
    0 [-2.6  7.5  0.4  8.9]
    1 [4.2 2.8 5.6 3.1]
    2 [ 8.9 -3.1  5.1 -3.6]
    3 [-2.2  4.9  1.1  7.4]
    3
    0 [-2.6  7.5  0.4  8.9]
    1 [4.2 2.8 5.6 3.1]
    2 [ 8.9 -3.1  5.1 -3.6]
    3 [-2.2  4.9  1.1  7.4]
    1
    0 [-2.6  7.5  0.4  8.9]
    1 [4.2 2.8 5.6 3.1]
    2 [ 8.9 -3.1  5.1 -3.6]
    3 [-2.2  4.9  1.1  7.4]
    1
    0 [-2.6  7.5  0.4  8.9]
    1 [4.2 2.8 5.6 3.1]
    2 [ 8.9 -3.1  5.1 -3.6]
    3 [-2.2  4.9  1.1  7.4]
    2
    0 [-2.6  7.5  0.4  8.9]
    1 [4.2 2.8 5.6 3.1]
    2 [ 8.9 -3.1  5.1 -3.6]
    3 [-2.2  4.9  1.1  7.4]
    2
    0 [-2.6  7.5  0.4  8.9]
    1 [4.2 2.8 5.6 3.1]
    2 [ 8.9 -3.1  5.1 -3.6]
    3 [-2.2  4.9  1.1  7.4]
    2
    0 [-2.6  7.5  0.4  8.9]
    1 [4.2 2.8 5.6 3.1]
    2 [ 8.9 -3.1  5.1 -3.6]
    3 [-2.2  4.9  1.1  7.4]
    3
    0 [-2.6  7.5  0.4  8.9]
    1 [4.2 2.8 5.6 3.1]
    2 [ 8.9 -3.1  5.1 -3.6]
    3 [-2.2  4.9  1.1  7.4]
    0
    0 [-2.6  7.5  0.4  8.9]
    1 [4.2 2.8 5.6 3.1]
    2 [ 8.9 -3.1  5.1 -3.6]
    3 [-2.2  4.9  1.1  7.4]
    0
    0 [-2.6  7.5  0.4  8.9]
    1 [4.2 2.8 5.6 3.1]
    2 [ 8.9 -3.1  5.1 -3.6]
    3 [-2.2  4.9  1.1  7.4]
    3
    0 [-2.6  7.5  0.4  8.9]
    1 [4.2 2.8 5.6 3.1]
    2 [ 8.9 -3.1  5.1 -3.6]
    3 [-2.2  4.9  1.1  7.4]
    2
    0 [-2.6  7.5  0.4  8.9]
    1 [4.2 2.8 5.6 3.1]
    2 [ 8.9 -3.1  5.1 -3.6]
    3 [-2.2  4.9  1.1  7.4]
    0
    0 [-2.6  7.5  0.4  8.9]
    1 [4.2 2.8 5.6 3.1]
    2 [ 8.9 -3.1  5.1 -3.6]
    3 [-2.2  4.9  1.1  7.4]
    0
    0 [-2.6  7.5  0.4  8.9]
    1 [4.2 2.8 5.6 3.1]
    2 [ 8.9 -3.1  5.1 -3.6]
    3 [-2.2  4.9  1.1  7.4]
    3
    0 [-2.6  7.5  0.4  8.9]
    1 [4.2 2.8 5.6 3.1]
    2 [ 8.9 -3.1  5.1 -3.6]
    3 [-2.2  4.9  1.1  7.4]
    0
    0 [-2.6  7.5  0.4  8.9]
    1 [4.2 2.8 5.6 3.1]
    2 [ 8.9 -3.1  5.1 -3.6]
    3 [-2.2  4.9  1.1  7.4]
    1
    0 [-2.6  7.5  0.4  8.9]
    1 [4.2 2.8 5.6 3.1]
    2 [ 8.9 -3.1  5.1 -3.6]
    3 [-2.2  4.9  1.1  7.4]
    1
    0 [-2.6  7.5  0.4  8.9]
    1 [4.2 2.8 5.6 3.1]
    2 [ 8.9 -3.1  5.1 -3.6]
    3 [-2.2  4.9  1.1  7.4]
    2
    0 [-2.6  7.5  0.4  8.9]
    1 [4.2 2.8 5.6 3.1]
    2 [ 8.9 -3.1  5.1 -3.6]
    3 [-2.2  4.9  1.1  7.4]
    2
    0 [-2.6  7.5  0.4  8.9]
    1 [4.2 2.8 5.6 3.1]
    2 [ 8.9 -3.1  5.1 -3.6]
    3 [-2.2  4.9  1.1  7.4]
    2
    0 [-2.6  7.5  0.4  8.9]
    1 [4.2 2.8 5.6 3.1]
    2 [ 8.9 -3.1  5.1 -3.6]
    3 [-2.2  4.9  1.1  7.4]
    0
    0 [-2.6  7.5  0.4  8.9]
    1 [4.2 2.8 5.6 3.1]
    2 [ 8.9 -3.1  5.1 -3.6]
    3 [-2.2  4.9  1.1  7.4]
    1
    0 [-2.6  7.5  0.4  8.9]
    1 [4.2 2.8 5.6 3.1]
    2 [ 8.9 -3.1  5.1 -3.6]
    3 [-2.2  4.9  1.1  7.4]
    3
    0 [-2.6  7.5  0.4  8.9]
    1 [4.2 2.8 5.6 3.1]
    2 [ 8.9 -3.1  5.1 -3.6]
    3 [-2.2  4.9  1.1  7.4]
    0
    0 [-2.6  7.5  0.4  8.9]
    1 [4.2 2.8 5.6 3.1]
    2 [ 8.9 -3.1  5.1 -3.6]
    3 [-2.2  4.9  1.1  7.4]
    0
    0 [-2.6  7.5  0.4  8.9]
    1 [4.2 2.8 5.6 3.1]
    2 [ 8.9 -3.1  5.1 -3.6]
    3 [-2.2  4.9  1.1  7.4]
    0
    0 [-2.6  7.5  0.4  8.9]
    1 [4.2 2.8 5.6 3.1]
    2 [ 8.9 -3.1  5.1 -3.6]
    3 [-2.2  4.9  1.1  7.4]
    2
    0 [-2.6  7.5  0.4  8.9]
    1 [4.2 2.8 5.6 3.1]
    2 [ 8.9 -3.1  5.1 -3.6]
    3 [-2.2  4.9  1.1  7.4]
    1
    0 [-2.6  7.5  0.4  8.9]
    1 [4.2 2.8 5.6 3.1]
    2 [ 8.9 -3.1  5.1 -3.6]
    3 [-2.2  4.9  1.1  7.4]
    2
    0 [-2.6  7.5  0.4  8.9]
    1 [4.2 2.8 5.6 3.1]
    2 [ 8.9 -3.1  5.1 -3.6]
    3 [-2.2  4.9  1.1  7.4]
    2
    0 [-2.6  7.5  0.4  8.9]
    1 [4.2 2.8 5.6 3.1]
    2 [ 8.9 -3.1  5.1 -3.6]
    3 [-2.2  4.9  1.1  7.4]
    1
    0 [-2.6  7.5  0.4  8.9]
    1 [4.2 2.8 5.6 3.1]
    2 [ 8.9 -3.1  5.1 -3.6]
    3 [-2.2  4.9  1.1  7.4]
    0
    0 [-2.6  7.5  0.4  8.9]
    1 [4.2 2.8 5.6 3.1]
    2 [ 8.9 -3.1  5.1 -3.6]
    3 [-2.2  4.9  1.1  7.4]
    0
    0 [-2.6  7.5  0.4  8.9]
    1 [4.2 2.8 5.6 3.1]
    2 [ 8.9 -3.1  5.1 -3.6]
    3 [-2.2  4.9  1.1  7.4]
    0
    0 [-2.6  7.5  0.4  8.9]
    1 [4.2 2.8 5.6 3.1]
    2 [ 8.9 -3.1  5.1 -3.6]
    3 [-2.2  4.9  1.1  7.4]
    0
    0 [-2.6  7.5  0.4  8.9]
    1 [4.2 2.8 5.6 3.1]
    2 [ 8.9 -3.1  5.1 -3.6]
    3 [-2.2  4.9  1.1  7.4]
    2
    0 [-2.6  7.5  0.4  8.9]
    1 [4.2 2.8 5.6 3.1]
    2 [ 8.9 -3.1  5.1 -3.6]
    3 [-2.2  4.9  1.1  7.4]
    2
    0 [-2.6  7.5  0.4  8.9]
    1 [4.2 2.8 5.6 3.1]
    2 [ 8.9 -3.1  5.1 -3.6]
    3 [-2.2  4.9  1.1  7.4]
    1
    0 [-2.6  7.5  0.4  8.9]
    1 [4.2 2.8 5.6 3.1]
    2 [ 8.9 -3.1  5.1 -3.6]
    3 [-2.2  4.9  1.1  7.4]
    1
    0 [-2.6  7.5  0.4  8.9]
    1 [4.2 2.8 5.6 3.1]
    2 [ 8.9 -3.1  5.1 -3.6]
    3 [-2.2  4.9  1.1  7.4]
    2
    0 [-2.6  7.5  0.4  8.9]
    1 [4.2 2.8 5.6 3.1]
    2 [ 8.9 -3.1  5.1 -3.6]
    3 [-2.2  4.9  1.1  7.4]
    2
    0 [-2.6  7.5  0.4  8.9]
    1 [4.2 2.8 5.6 3.1]
    2 [ 8.9 -3.1  5.1 -3.6]
    3 [-2.2  4.9  1.1  7.4]
    0
    0 [-2.6  7.5  0.4  8.9]
    1 [4.2 2.8 5.6 3.1]
    2 [ 8.9 -3.1  5.1 -3.6]
    3 [-2.2  4.9  1.1  7.4]
    0
    0 [-2.6  7.5  0.4  8.9]
    1 [4.2 2.8 5.6 3.1]
    2 [ 8.9 -3.1  5.1 -3.6]
    3 [-2.2  4.9  1.1  7.4]
    0
    0 [-2.6  7.5  0.4  8.9]
    1 [4.2 2.8 5.6 3.1]
    2 [ 8.9 -3.1  5.1 -3.6]
    3 [-2.2  4.9  1.1  7.4]
    2
    0 [-2.6  7.5  0.4  8.9]
    1 [4.2 2.8 5.6 3.1]
    2 [ 8.9 -3.1  5.1 -3.6]
    3 [-2.2  4.9  1.1  7.4]
    3
    0 [-2.6  7.5  0.4  8.9]
    1 [4.2 2.8 5.6 3.1]
    2 [ 8.9 -3.1  5.1 -3.6]
    3 [-2.2  4.9  1.1  7.4]
    2
    0 [-2.6  7.5  0.4  8.9]
    1 [4.2 2.8 5.6 3.1]
    2 [ 8.9 -3.1  5.1 -3.6]
    3 [-2.2  4.9  1.1  7.4]
    0
    0 [-2.6  7.5  0.4  8.9]
    1 [4.2 2.8 5.6 3.1]
    2 [ 8.9 -3.1  5.1 -3.6]
    3 [-2.2  4.9  1.1  7.4]
    0
    0 [-2.6  7.5  0.4  8.9]
    1 [4.2 2.8 5.6 3.1]
    2 [ 8.9 -3.1  5.1 -3.6]
    3 [-2.2  4.9  1.1  7.4]
    0
    0 [-2.6  7.5  0.4  8.9]
    1 [4.2 2.8 5.6 3.1]
    2 [ 8.9 -3.1  5.1 -3.6]
    3 [-2.2  4.9  1.1  7.4]
    2
    0 [-2.6  7.5  0.4  8.9]
    1 [4.2 2.8 5.6 3.1]
    2 [ 8.9 -3.1  5.1 -3.6]
    3 [-2.2  4.9  1.1  7.4]
    1
    0 [-2.6  7.5  0.4  8.9]
    1 [4.2 2.8 5.6 3.1]
    2 [ 8.9 -3.1  5.1 -3.6]
    3 [-2.2  4.9  1.1  7.4]
    0
    0 [-2.6  7.5  0.4  8.9]
    1 [4.2 2.8 5.6 3.1]
    2 [ 8.9 -3.1  5.1 -3.6]
    3 [-2.2  4.9  1.1  7.4]
    2
    0 [-2.6  7.5  0.4  8.9]
    1 [4.2 2.8 5.6 3.1]
    2 [ 8.9 -3.1  5.1 -3.6]
    3 [-2.2  4.9  1.1  7.4]
    3
    0 [-2.6  7.5  0.4  8.9]
    1 [4.2 2.8 5.6 3.1]
    2 [ 8.9 -3.1  5.1 -3.6]
    3 [-2.2  4.9  1.1  7.4]
    0
    0 [-2.6  7.5  0.4  8.9]
    1 [4.2 2.8 5.6 3.1]
    2 [ 8.9 -3.1  5.1 -3.6]
    3 [-2.2  4.9  1.1  7.4]
    2
    0 [-2.6  7.5  0.4  8.9]
    1 [4.2 2.8 5.6 3.1]
    2 [ 8.9 -3.1  5.1 -3.6]
    3 [-2.2  4.9  1.1  7.4]
    0
    0 [-2.6  7.5  0.4  8.9]
    1 [4.2 2.8 5.6 3.1]
    2 [ 8.9 -3.1  5.1 -3.6]
    3 [-2.2  4.9  1.1  7.4]
    2
    0 [-2.6  7.5  0.4  8.9]
    1 [4.2 2.8 5.6 3.1]
    2 [ 8.9 -3.1  5.1 -3.6]
    3 [-2.2  4.9  1.1  7.4]
    1
    0 [-2.6  7.5  0.4  8.9]
    1 [4.2 2.8 5.6 3.1]
    2 [ 8.9 -3.1  5.1 -3.6]
    3 [-2.2  4.9  1.1  7.4]
    1
    0 [-2.6  7.5  0.4  8.9]
    1 [4.2 2.8 5.6 3.1]
    2 [ 8.9 -3.1  5.1 -3.6]
    3 [-2.2  4.9  1.1  7.4]
    3
    0 [-2.6  7.5  0.4  8.9]
    1 [4.2 2.8 5.6 3.1]
    2 [ 8.9 -3.1  5.1 -3.6]
    3 [-2.2  4.9  1.1  7.4]
    0
    0 [-2.6  7.5  0.4  8.9]
    1 [4.2 2.8 5.6 3.1]
    2 [ 8.9 -3.1  5.1 -3.6]
    3 [-2.2  4.9  1.1  7.4]
    0
    0 [-2.6  7.5  0.4  8.9]
    1 [4.2 2.8 5.6 3.1]
    2 [ 8.9 -3.1  5.1 -3.6]
    3 [-2.2  4.9  1.1  7.4]
    3
    0 [-2.6  7.5  0.4  8.9]
    1 [4.2 2.8 5.6 3.1]
    2 [ 8.9 -3.1  5.1 -3.6]
    3 [-2.2  4.9  1.1  7.4]
    0
    0 [-2.6  7.5  0.4  8.9]
    1 [4.2 2.8 5.6 3.1]
    2 [ 8.9 -3.1  5.1 -3.6]
    3 [-2.2  4.9  1.1  7.4]
    0
    0 [-2.6  7.5  0.4  8.9]
    1 [4.2 2.8 5.6 3.1]
    2 [ 8.9 -3.1  5.1 -3.6]
    3 [-2.2  4.9  1.1  7.4]
    0
    0 [-2.6  7.5  0.4  8.9]
    1 [4.2 2.8 5.6 3.1]
    2 [ 8.9 -3.1  5.1 -3.6]
    3 [-2.2  4.9  1.1  7.4]
    2
    0 [-2.6  7.5  0.4  8.9]
    1 [4.2 2.8 5.6 3.1]
    2 [ 8.9 -3.1  5.1 -3.6]
    3 [-2.2  4.9  1.1  7.4]
    0
    0 [-2.6  7.5  0.4  8.9]
    1 [4.2 2.8 5.6 3.1]
    2 [ 8.9 -3.1  5.1 -3.6]
    3 [-2.2  4.9  1.1  7.4]
    2
    0 [-2.6  7.5  0.4  8.9]
    1 [4.2 2.8 5.6 3.1]
    2 [ 8.9 -3.1  5.1 -3.6]
    3 [-2.2  4.9  1.1  7.4]
    1
    0 [-2.6  7.5  0.4  8.9]
    1 [4.2 2.8 5.6 3.1]
    2 [ 8.9 -3.1  5.1 -3.6]
    3 [-2.2  4.9  1.1  7.4]
    1


    /usr/local/lib/python3.6/dist-packages/numpy/core/fromnumeric.py:3335: RuntimeWarning: Mean of empty slice.
      out=out, **kwargs)
    /usr/local/lib/python3.6/dist-packages/numpy/core/_methods.py:161: RuntimeWarning: invalid value encountered in double_scalars
      ret = ret.dtype.type(ret / rcount)



    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-463-ceae4e019929> in <module>()
         18         #print(i,clusters[i],'\n\n')
         19         #new_cc[i] = list(np.around(np.sum(clusters[i],axis=0)/len(clusters[i]),3))
    ---> 20         new_cc[i] = list(np.around(np.mean(clusters[i],axis=0),3))
         21 
         22     print("new_cc", new_cc)


    TypeError: 'numpy.float64' object is not iterable



```
print(consine_distance([ 0.66,  4.5,   2.02,  1.38],[  3.2,   5.5,  -6.6, -11.2]))
```

    1.02576676340951



```

```


```
print(cos_distance(np.array([ 0.66,  4.5,   2.02,  1.38]),np.array([  3.2,   5.5,  -6.6, -11.2])))
```

    1.02576676340951



```
cc=[ 0.66,  4.5,   2.02,  1.38]
gvPt=[  3.2,   5.5,  -6.6, -11.2]
```


```
getClusterCentre(cc, gvPt)
```

    0 0.66



    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-411-2ff5d6c99298> in <module>()
    ----> 1 getClusterCentre(cc, gvPt)
    

    <ipython-input-404-9a513c398280> in getClusterCentre(gvCenters, gvPt)
          7         print(i,cc)
          8         #tmp_dist = math.sqrt(pow(cc[0] - gvPt[0], 2) + pow(cc[1] - gvPt[1], 2))
    ----> 9         tmp_dist = consine_distance(cc,gvPt)
         10         if(tmp_dist < min_distance):
         11             min_distance = tmp_dist


    <ipython-input-397-31dc2b8bfeec> in consine_distance(a, b)
          1 def consine_distance(a,b):
          2   d_tmp=np.dot(a,b)
    ----> 3   a_tmp=math.sqrt(sum(np.square(a)))
          4   b_tmp=math.sqrt(sum(np.square(b)))
          5   d=1-(d_tmp/(a_tmp*b_tmp))


    TypeError: 'numpy.float64' object is not iterable



```
cc@gvPt
```




    -1.9259999999999948




```
cc.dot(gvPt)
```




    -1.9259999999999948




```
np.dot(cc,gvPt)
```




    -1.9259999999999948




```
np.linalg.norm(cc)
```




    5.164339260738009




```
np.linalg.norm(gvPt)
```




    14.473769377739856




```
# With numpy
import math
def cos_distance(a,b):
  return 1-(a@b)/( np.linalg.norm(a)*np.linalg.norm(b) )
# assume data is 2D
def getClusterCentre(gvCenters, gvPt):
    min_distance = 100000
    i_centre = -1
    for i,cc in enumerate(gvCenters):
        tmp_dist = cos_distance(cc, gvPt) 
        if(tmp_dist < min_distance):
            min_distance = tmp_dist
            i_centre = i
    return i_centre
```

#hierarchical clustering


```
def ecludian_distance(A,B):
  A=np.array(A)
  B=np.array(B)
  return np.around(np.sqrt(np.sum((A - B)**2)),3)
```


```
def manhattan_distance(A,B):
  A=np.array(A)
  B=np.array(B)
  return np.around(np.abs(A - B).sum(),3)
```


```

```


```
manhattan_distance([1,2,5,6],[2,3,5,5])
```




    3




```
A=np.array([1,2,5,6])
B=np.array([2,3,5,5])
```


```
manhattan_distance(A,B)
```

    <class 'numpy.ndarray'>
    <class 'numpy.ndarray'>





    3




```
round(math.sqrt(pow(A[0] - B[0], 2) + pow(A[1] - B[1], 2)),3)
```




    1.414



#GMM


```
import numpy as np
from scipy.stats import multivariate_normal

class GMM:
    def __init__(self, k, max_iter=5):
        self.k = k
        self.max_iter = int(max_iter)

    def initialize(self, X):
        self.shape = X.shape
        self.n, self.m = self.shape

        self.phi = np.full(shape=self.k, fill_value=1/self.k)
        self.weights = np.full( shape=self.shape, fill_value=1/self.k)
        
        random_row = np.random.randint(low=0, high=self.n, size=self.k)
        self.mu = [  X[row_index,:] for row_index in random_row ]
        self.sigma = [ np.cov(X.T) for _ in range(self.k) ]

    def e_step(self, X):
        # E-Step: update weights and phi holding mu and sigma constant
        self.weights = self.predict_proba(X)
        self.phi = self.weights.mean(axis=0)
    
    def m_step(self, X):
        # M-Step: update mu and sigma holding phi and weights constant
        for i in range(self.k):
            weight = self.weights[:, [i]]
            total_weight = weight.sum()
            self.mu[i] = (X * weight).sum(axis=0) / total_weight
            self.sigma[i] = np.cov(X.T, 
                aweights=(weight/total_weight).flatten(), 
                bias=True)

    def fit(self, X):
        self.initialize(X)
        
        for iteration in range(self.max_iter):
            self.e_step(X)
            self.m_step(X)
            
    def predict_proba(self, X):
        likelihood = np.zeros( (self.n, self.k) )
        for i in range(self.k):
            distribution = multivariate_normal(
                mean=self.mu[i], 
                cov=self.sigma[i])
            likelihood[:,i] = distribution.pdf(X)
        
        numerator = likelihood * self.phi
        denominator = numerator.sum(axis=1)[:, np.newaxis]
        weights = numerator / denominator
        return weights
    
    def predict(self, X):
        weights = self.predict_proba(X)
        return np.argmax(weights, axis=1)
```


```

```


```
from scipy.stats import mode
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
```


```
np.random.seed(42)
gmm = GMM(k=3, max_iter=10)
gmm.fit(X)
```


```
permutation = np.array([
    mode(iris.target[gmm.predict(X) == i]).mode.item() 
    for i in range(gmm.k)])
permuted_prediction = permutation[gmm.predict(X)]
print(np.mean(iris.target == permuted_prediction))
confusion_matrix(iris.target, permuted_prediction)
```

    0.96





    array([[50,  0,  0],
           [ 0, 44,  6],
           [ 0,  0, 50]])




```
def jitter(x):
    return x + np.random.uniform(low=-0.05, high=0.05, size=x.shape)

def plot_axis_pairs(X, axis_pairs, clusters, classes):
    n_rows = len(axis_pairs) // 2
    n_cols = 2
    plt.figure(figsize=(16, 10))
    for index, (x_axis, y_axis) in enumerate(axis_pairs):
        plt.subplot(n_rows, n_cols, index+1)
        plt.title('GMM Clusters')
        plt.xlabel(iris.feature_names[x_axis])
        plt.ylabel(iris.feature_names[y_axis])
        plt.scatter(
            jitter(X[:, x_axis]), 
            jitter(X[:, y_axis]), 
            #c=clusters, 
            cmap=plt.cm.get_cmap('brg'),
            marker='x')
    plt.tight_layout()
    
plot_axis_pairs(
    X=X,
    axis_pairs=[ 
        (0,1), (2,3), 
        (0,2), (1,3) ],
    clusters=permuted_prediction,
    classes=iris.target)
```


![png](assets/practice_Pad_146_0.png)


#Gaussian_Mixture_Models 1D data


```
import matplotlib.pyplot as plt
from sklearn import cluster, datasets, mixture
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.datasets import make_spd_matrix
plt.rcParams["axes.grid"] = False
```


```
# example of a bimodal constructed from two gaussian processes
from numpy import hstack
from numpy.random import normal
from matplotlib import pyplot
# generate a sample
X1 = normal(loc=20, scale=5, size=3000)
X2 = normal(loc=40, scale=5, size=7000)
X = hstack((X1, X2))

X3 = normal(loc=60, scale=5, size=2000)
X = hstack((X1, X2, X3))
# plot the histogram
pyplot.hist(X, bins=50, density=True)
pyplot.show()
```


![png](assets/practice_Pad_149_0.png)



```
def pdf(data, mean: float, variance: float):
  # A normal continuous random variable.
  s1 = 1/(np.sqrt(2*np.pi*variance))
  s2 = np.exp(-(np.square(data - mean)/(2*variance)))
  return s1 * s2
```


```
# define the number of clusters to be learned and initialize parameters
k = 3
weights = np.ones((k)) / k
means = np.random.choice(X, k)
variances = np.random.random_sample(size=k)
print(weights,means, variances)
```

    [0.33333333 0.33333333 0.33333333] [38.98652447 21.44861724 13.74031401] [0.11924318 0.5789163  0.3234807 ]



```
X = np.array(X)
print(X.shape)
```

    (12000,)



```
t=1e-8
for step in range(25):

  likelihood = []
  

  # Expectation step
  for j in range(k):
    likelihood.append(pdf(X, means[j], np.sqrt(variances[j])))
  likelihood = np.array(likelihood)
    
  b = []
  # Maximization step 
  for j in range(k):
    # use the current values for the parameters to evaluate the posterior
    # probabilities of the data to have been generanted by each gaussian    
    b.append((likelihood[j] * weights[j]) / (np.sum([likelihood[i] * weights[i] for i in range(k)], axis=0)+t))
  
    # updage mean and variance
    means[j] = np.sum(b[j] * X) / (np.sum(b[j]+t))
    variances[j] = np.sum(b[j] * np.square(X - means[j])) / (np.sum(b[j]+t))
    
    # update the weights
    weights[j] = np.mean(b[j])
  print(means,variances)
```


```
# visualize the training data
bins = np.linspace(np.min(X),np.max(X),100)

plt.figure(figsize=(10,7))
plt.xlabel("$x$")
plt.ylabel("pdf")
plt.scatter(X, [0.00005] * len(X), color='red', s=30, marker=2, label="Trained data")
#plt.hist(X, bins=100, density=True,label="Trained data")
plt.plot(bins, pdf(bins, means[0], variances[0]), color='blue', label="Cluster 1")
plt.plot(bins, pdf(bins, means[1], variances[1]), color='green', label="Cluster 2")
plt.plot(bins, pdf(bins, means[2], variances[1]), color='red', label="Cluster 3")
#plt.plot(bins, pdf(bins, mu3, sigma3), color='red')

plt.legend()
plt.plot()
```




    []




![png](assets/practice_Pad_154_1.png)


#Multivariate Classifcation

##Multivariate k=2 class problem on 2-d data


```
import numpy as np
import pandas as pd
import random
import math
```


```
center1_x = 1
center1_y = 2

center2_x = 2
center2_y = 3

centers_dict = {}
centers_dict[(center1_x,center1_y)] = "class1"
centers_dict[(center2_x,center2_y)] = "class2"
centers_dict
```




    {(1, 2): 'class1', (2, 3): 'class2'}




```
N = 40
np.random.seed(0)
sample_x_1 = np.round(np.random.uniform(center1_x-1, center1_x+1, N), 3)
sample_x_2 = np.round(np.random.uniform(center2_x-1, center2_x+1, N), 3)

sample_y_1 = np.round(np.random.uniform(center1_y-1, center1_y+1, N), 3)
sample_y_2 = np.round(np.random.uniform(center2_y-1, center2_y+1, N), 3)
```


```
x_orig = sample_x_1 + sample_x_2
y_orig = sample_y_1 + sample_y_2

r1_orig = [1]*int(N/2) + [0]*int(N/2)
r2_orig = [0]*int(N/2) + [1]*int(N/2)

print('dataset')
print(list(zip(x_orig, y_orig, r1_orig, r2_orig)))
```

    dataset
    [(2.817, 5.087, 1, 0), (3.3040000000000003, 4.832, 1, 0), (3.601, 5.04, 1, 0), (2.21, 5.673, 1, 0), (3.181, 4.981, 1, 0), (3.633, 4.744, 1, 0), (2.2960000000000003, 4.084, 1, 0), (3.042, 3.7910000000000004, 1, 0), (3.558, 5.4719999999999995, 1, 0), (2.494, 5.439, 1, 0), (3.723, 4.873, 1, 0), (2.935, 5.193, 1, 0), (4.1129999999999995, 3.535, 1, 0), (3.0549999999999997, 5.029999999999999, 1, 0), (1.5599999999999998, 4.719, 1, 0), (1.4969999999999999, 4.548, 1, 0), (2.346, 5.322, 1, 0), (3.1719999999999997, 4.346, 1, 0), (3.489, 5.962, 1, 0), (3.229, 3.872, 1, 0), (3.2750000000000004, 6.149, 0, 1), (2.819, 4.275, 0, 1), (3.236, 5.3420000000000005, 0, 1), (2.8369999999999997, 6.708, 0, 1), (1.63, 5.11, 0, 1), (3.0170000000000003, 5.5600000000000005, 0, 1), (2.929, 4.384, 0, 1), (3.083, 5.984, 0, 1), (3.72, 4.874, 0, 1), (2.021, 6.9030000000000005, 0, 1), (3.4819999999999998, 4.193, 0, 1), (3.4850000000000003, 6.429, 0, 1), (3.866, 4.724, 0, 1), (3.347, 4.826, 0, 1), (2.517, 4.876, 0, 1), (2.313, 5.489, 0, 1), (2.79, 6.377000000000001, 0, 1), (2.474, 5.301, 0, 1), (3.479, 5.577, 0, 1), (2.601, 4.523, 0, 1)]



```
#shuffle
x = []
y = []
r1 = []
r2 = []
shuffled_indices = list(range(len(x_orig)))
random.shuffle(shuffled_indices)

for i in shuffled_indices:
    x.append(x_orig[i])
    y.append(y_orig[i])
    r1.append(r1_orig[i])
    r2.append(r2_orig[i])
```


```
i_trSetSize = int(0.8*N)
i_testSetSize = N - i_trSetSize

x_train = x[0:i_trSetSize]
y_train = y[0:i_trSetSize]
r1_train = r1[0:i_trSetSize]
r2_train = r2[0:i_trSetSize]

x_test = x[i_trSetSize+1:]
y_test = y[i_trSetSize+1:]
r1_test = r1[i_trSetSize+1:]
r2_test = r2[i_trSetSize+1:]
```


```
x_train
```




    [2.517,
     2.8369999999999997,
     2.601,
     2.21,
     1.63,
     3.1719999999999997,
     3.236,
     2.819,
     3.558,
     3.633,
     3.866,
     3.0549999999999997,
     2.021,
     3.229,
     2.494,
     3.4819999999999998,
     3.0170000000000003,
     3.72,
     3.181,
     1.4969999999999999,
     3.723,
     3.479,
     3.601,
     3.042,
     2.929,
     2.935,
     2.474,
     2.2960000000000003,
     2.817,
     1.5599999999999998,
     3.2750000000000004,
     2.313]


