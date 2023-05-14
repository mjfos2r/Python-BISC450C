# Python Fundamentals


```python
# Any python interpreter can be used as a calculator: 
3 + 5 * 4
```




    23




```python
# lets set a variable to a value
weight_kg = 60
```


```python
print(weight_kg)
```

    60



```python
# weight0 = valid
# 0weight = invalid
# weight and Weight are different
```


```python
# Types of data
# three common types of data
## Integers
## floating point numbers
## strings
```


```python
# floating point number
weight_kg = 60.3

# string of letters
patient_name = 'John Smith'

#String comprised of numbers
patient_id = '001'
```


```python
# use variables in python

# convert weight in kg to lbs
weight_lb = 2.2 * weight_kg

print(weight_lb)
```

    132.66



```python
# can add to strings
#lets add a prefix to patient ID
patient_id = 'inflam_' + patient_id
print(patient_id)
```

    inflam_001



```python
# lets combine print statements
print(patient_id, 'weight in kg:', weight_kg)
```

    inflam_001 weight in kg: 60.3



```python
# we can call a function inside another function
print(type(60.3))
print(type(patient_id))
```

    <class 'float'>
    <class 'str'>



```python
# we can also do calculations inside the print function
print('weight in lbs:', 2.2 * weight_kg)
```

    weight in lbs: 132.66



```python
print(weight_kg)
```

    60.3



```python
weight_kg = 65.0
print('weight in kilograms is now:', weight_kg)
```

    weight in kilograms is now: 65.0

