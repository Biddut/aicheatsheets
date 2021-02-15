# Statistics

**In  linear regression, there may be collinearity between the independent variables. In logistic regression, there should not be collinearity between the independent variable. Why?**

In linear regression, we assume that independent variables have a distinct impact on the dependent variable. In other words, we can say X-variable will contain unique pieces of information for about.

For example :  Assume a multiple regression model is: 

![](../../.gitbook/assets/image%20%281%29.png)

We believe that, 

* B1= the change in Y for a 1-unit change in X1, while X2 held constant
* B2= the change in Y for a 1-unit change in X2, while X1 held constant

if the above conditions not satisfied by the model then we can say the model has multicollinearity.

#### Effects of Multicollinearity

1. Variances \( and standard errors\) of regression co-efficient estimators are inflated. This means that Var\(bi\) is too large.
2. The magnitude of the bi may be different from what we expected
3. The signs of bi may become opposite than expected. 
4. Adding or removing any of the  X-variables produces a large difference in the value of remaining bi or their signs. 
5. In some cases, F is significant, but t-values may not be significant 

#### Test of Multicollinearity

1. Calculate the correlation coefficient \(r\) for each pair of the X-variables. If the any of r-value significantly differ from zero, there might have a collinearity in the pair.

![](../../.gitbook/assets/image%20%282%29.png)

* Caveat:  Although the r of any two variables may be too small, three independent variable \(x1,x2, x3\)  may be highly correlated with each other. 

 2. Check if the Variation Inflation Factor \(VIF\) is too high.

![](../../.gitbook/assets/image%20%283%29.png)

* Rule of thumbs for VIF:
  *  If VIF&gt;5  , colinearity exists. 
  * If VIF=1 , no collinearity 

![](../../.gitbook/assets/image%20%284%29.png)

####  Solutions of Multicollinearity 

* * [ ] [https://statisticalhorizons.com/multicollinearity](https://statisticalhorizons.com/multicollinearity)
* [ ] [https://statisticsbyjim.com/regression/multicollinearity-in-regression-analysis/](https://statisticsbyjim.com/regression/multicollinearity-in-regression-analysis/)
* [ ] [https://www.youtube.com/watch?v=NAPhUDjgG\_s](https://www.youtube.com/watch?v=NAPhUDjgG_s)
* [ ] [https://www.youtube.com/watch?v=pZhm1GMn2GY](https://www.youtube.com/watch?v=pZhm1GMn2GY)



  **2. What is the Central Limit Theorem? Explain it. Why is it important?**

## How many and what type of Data need to deal with machine learning?

Yes, after a few months we finally found the answer. Sadly, Mike is on vacations right now so I'm afraid we are not able to provide the answer at this point.

* [https://medium.com/swlh/data-types-in-statistics-used-for-machine-learning-5b4c24ae6036](https://medium.com/swlh/data-types-in-statistics-used-for-machine-learning-5b4c24ae6036)
* [https://towardsdatascience.com/7-data-types-a-better-way-to-think-about-data-types-for-machine-learning-939fae99a689](https://towardsdatascience.com/7-data-types-a-better-way-to-think-about-data-types-for-machine-learning-939fae99a689)





