```python
a=input("this number is:")
print(a)#input返回值为字符串，例"1212"
```

    this number is:1212
    1212
    


```python
a=int(input("this number is:"))
print(a)#此时转换成整数
```

    this number is:444
    444
    
input函数返回值为字符串，故需要用整数型时需要强制转换
# 输出时格式化参数的使用


```python
a=int (input("请输入整数"))
print("刚才输入的整数为：%d"%(a))
```

    请输入整数111111
    刚才输入的整数为：111111
    

# 失效转义字符
失效转义字符用r或R

```python
print(r"E:\r\daima\n")
```

    E:\r\daima\n
    


```python
print(R"E:\r\daima\n")
```

    E:\r\daima\n
    

# 输出时的end参数
print函数为print（输出内容，end），end默认为\n，可以修改

```python
print("adf")
print("ffff")
```

    adf
    ffff
    


```python
print("adf",end="")
print("fffff")
```

    adffffff
    


```python
print("adf",end=" ")
print("fffff")
```

    adf fffff
    


```python

```
