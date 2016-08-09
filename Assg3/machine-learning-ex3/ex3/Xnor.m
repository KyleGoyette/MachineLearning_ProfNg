clear;close all;clc

X=[1 0 0]

Theta1=[-30 20 20]
Theta2=[10 -20 -20]
Theta3=[-10 20 20]
Z11=sigmoid(Theta1*X')
Z12=sigmoid(Theta2*X')
Z=[1 Z11 Z12]
Z3=sigmoid(Theta3*Z')