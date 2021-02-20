# Basic Statistics

1. **In linear regression, there may be collinearity between the independent variables. In logistic regression, there should not be collinearity between the independent variable. Why?**

In linear regression, we assume that independent variables have a distinct impact on the dependent variable. In other words, we can say the independent variables will contain unique pieces of information for the dependent variable.

For example: Assume a multiple regression model is:

![](.gitbook/assets/image%20%281%29.png)

We believe that,

* B1= the change in Y for a 1-unit change in X1, while X2 held constant
* B2= the change in Y for a 1-unit change in X2, while X1 held constant

if the above conditions do not satisfy by the model then we can say the model has a multicollinearity issue.

### Effects of Having Multicollinearity:

1. Variances \(and standard errors\) of regression co-efficient estimators are inflated. This means that Var`(bi)` is too large.
2. The magnitude of the `bi` maybe different from what we expected
3. The signs of `bi` may become opposite than expected. 
4. Adding or removing any of X-variables produces a large difference in the value of remaining bi or their signs. 

#### Helper Links :

1. [ ] [https://statisticalhorizons.com/multicollinearity](https://statisticalhorizons.com/multicollinearity)
2. [ ] [https://statisticsbyjim.com/regression/multicollinearity-in-regression-analysis/](https://statisticsbyjim.com/regression/multicollinearity-in-regression-analysis/)
3. [ ] [https://www.youtube.com/watch?v=NAPhUDjgG\_s](https://www.youtube.com/watch?v=NAPhUDjgG_s)
4. [ ] [https://www.youtube.com/watch?v=Cba9LJ9lS8s](https://www.youtube.com/watch?v=Cba9LJ9lS8s)

## 

