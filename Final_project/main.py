#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tkinter as tk
from tkinter import messagebox
import numpy as np
from LU_decomposition import*
from scipy.linalg import lu

matrix_A=np.zeros((3,3))

def EntryA_get1():
    return float(entryA_1.get()),float(entryA_2.get()),float(entryA_3.get())
def EntryA_get2():
    return float(entryA_4.get()),float(entryA_5.get()),float(entryA_6.get())
def EntryA_get3():
    return float(entryA_7.get()),float(entryA_8.get()),float(entryA_9.get())
    
def determinant_A():
    frame_3=tk.Frame(win,bg="light blue",width=900,height=200).grid(row=2,column=0)
    matrix_A[0][0],matrix_A[0][1],matrix_A[0][2]=EntryA_get1()
    matrix_A[1][0],matrix_A[1][1],matrix_A[1][2]=EntryA_get2()
    matrix_A[2][0],matrix_A[2][1],matrix_A[2][2]=EntryA_get3()
    det_A=np.linalg.det(matrix_A)
    label_3=tk.Label(frame_3,text="行列式: {}".format(det_A),font=('Arial',16),bg="light blue")
    label_3.place(x=50,y=430)
def transpose_A():
    frame_3=tk.Frame(win,bg="light blue",width=900,height=200).grid(row=2,column=0)
    matrix_A[0][0],matrix_A[0][1],matrix_A[0][2]=EntryA_get1()
    matrix_A[1][0],matrix_A[1][1],matrix_A[1][2]=EntryA_get2()
    matrix_A[2][0],matrix_A[2][1],matrix_A[2][2]=EntryA_get3()
    A_T=matrix_A.T
    label_3=tk.Label(frame_3,text="轉置矩陣:\n{}".format(A_T),font=('Arial',16),bg="light blue")
    label_3.place(x=50,y=430)
def inverse_A():
    frame_3=tk.Frame(win,bg="light blue",width=900,height=200).grid(row=2,column=0)
    matrix_A[0][0],matrix_A[0][1],matrix_A[0][2]=EntryA_get1()
    matrix_A[1][0],matrix_A[1][1],matrix_A[1][2]=EntryA_get2()
    matrix_A[2][0],matrix_A[2][1],matrix_A[2][2]=EntryA_get3()
    if np.linalg.det(matrix_A)==0:
        messagebox.showerror("Error","行列式為零，此矩陣不可逆")
    else:
        A_inv=np.linalg.inv(matrix_A)
        label_3=tk.Label(frame_3,text="逆矩陣:\n{}".format(A_inv),font=('Arial',16),bg="light blue")
        label_3.place(x=50,y=430)
def rank_A():
    frame_3=tk.Frame(win,bg="light blue",width=900,height=200).grid(row=2,column=0)
    matrix_A[0][0],matrix_A[0][1],matrix_A[0][2]=EntryA_get1()
    matrix_A[1][0],matrix_A[1][1],matrix_A[1][2]=EntryA_get2()
    matrix_A[2][0],matrix_A[2][1],matrix_A[2][2]=EntryA_get3()
    A_rank=np.linalg.matrix_rank(matrix_A)
    label_3=tk.Label(frame_3,text="秩: {}".format(A_rank),font=('Arial',16),bg="light blue")
    label_3.place(x=50,y=430)
def LUdecoposition_A():
    frame_3=tk.Frame(win,bg="light blue",width=900,height=200).grid(row=2,column=0)
    matrix_A[0][0],matrix_A[0][1],matrix_A[0][2]=EntryA_get1()
    matrix_A[1][0],matrix_A[1][1],matrix_A[1][2]=EntryA_get2()
    matrix_A[2][0],matrix_A[2][1],matrix_A[2][2]=EntryA_get3()
    if np.linalg.det(matrix_A)==0:
        messagebox.showerror("Error","行列式為零，此矩陣無法做LU分解")
    elif matrix_A[0][0]==0:
        messagebox.showwarning("Warning","請做PLU分解")
    else:
        U,L=LU_decomposition(matrix_A)
        label_3=tk.Label(frame_3,text="L:\n{}".format(L),font=('Arial',14),bg="light blue")
        label_3.place(x=50,y=430)
        label_4=tk.Label(frame_3,text="U:\n{}".format(U),font=('Arial',14),bg="light blue")
        label_4.place(x=470,y=430)
def PLUdecoposition_A():
    frame_3=tk.Frame(win,bg="light blue",width=900,height=200).grid(row=2,column=0)
    matrix_A[0][0],matrix_A[0][1],matrix_A[0][2]=EntryA_get1()
    matrix_A[1][0],matrix_A[1][1],matrix_A[1][2]=EntryA_get2()
    matrix_A[2][0],matrix_A[2][1],matrix_A[2][2]=EntryA_get3()
    P, L, U = lu(matrix_A)
    label_3=tk.Label(frame_3,text="P:\n{}".format(P),font=('Arial',12),bg="light blue")
    label_3.place(x=50,y=430)
    label_4=tk.Label(frame_3,text="L:\n{}".format(L),font=('Arial',12),bg="light blue")
    label_4.place(x=250,y=430)
    label_5=tk.Label(frame_3,text="U:\n{}".format(U),font=('Arial',12),bg="light blue")
    label_5.place(x=600,y=430)
def Choleskydecoposition_A():
    frame_3=tk.Frame(win,bg="light blue",width=900,height=200).grid(row=2,column=0)
    matrix_A[0][0],matrix_A[0][1],matrix_A[0][2]=EntryA_get1()
    matrix_A[1][0],matrix_A[1][1],matrix_A[1][2]=EntryA_get2()
    matrix_A[2][0],matrix_A[2][1],matrix_A[2][2]=EntryA_get3()
    if (matrix_A==matrix_A.T).all()==False:
        messagebox.showerror("Error","因為A≠A^T，無法做Cholesky分解")
    elif np.all(np.linalg.eigvals(matrix_A) > 0)==False:
        messagebox.showerror("Error","非正定矩陣，無法做Cholesky分解")
    else:
        L = np.linalg.cholesky(matrix_A)
        label_3=tk.Label(frame_3,text="Cholesky分解:\n{}".format(L),font=('Arial',14),bg="light blue")
        label_3.place(x=50,y=430)
        label_4=tk.Label(frame_3,text="\n{}".format(L.T),font=('Arial',14),bg="light blue")
        label_4.place(x=470,y=430)
    


# In[2]:


matrix_B=np.zeros((3,3))

def EntryB_get1():
    return float(entryB_1.get()),float(entryB_2.get()),float(entryB_3.get())
def EntryB_get2():
    return float(entryB_4.get()),float(entryB_5.get()),float(entryB_6.get())
def EntryB_get3():
    return float(entryB_7.get()),float(entryB_8.get()),float(entryB_9.get())
    
def determinant_B():
    frame_3=tk.Frame(win,bg="light blue",width=900,height=200).grid(row=2,column=0)
    matrix_B[0][0],matrix_B[0][1],matrix_B[0][2]=EntryB_get1()
    matrix_B[1][0],matrix_B[1][1],matrix_B[1][2]=EntryB_get2()
    matrix_B[2][0],matrix_B[2][1],matrix_B[2][2]=EntryB_get3()
    det_B=np.linalg.det(matrix_B)
    label_3=tk.Label(frame_3,text="行列式: {}".format(det_B),font=('Arial',16),bg="light blue")
    label_3.place(x=50,y=430)
def transpose_B():
    frame_3=tk.Frame(win,bg="light blue",width=900,height=200).grid(row=2,column=0)
    matrix_B[0][0],matrix_B[0][1],matrix_B[0][2]=EntryB_get1()
    matrix_B[1][0],matrix_B[1][1],matrix_B[1][2]=EntryB_get2()
    matrix_B[2][0],matrix_B[2][1],matrix_B[2][2]=EntryB_get3()
    B_T=matrix_B.T
    label_3=tk.Label(frame_3,text="轉置矩陣:\n{}".format(B_T),font=('Arial',16),bg="light blue")
    label_3.place(x=50,y=430)
def inverse_B():
    frame_3=tk.Frame(win,bg="light blue",width=900,height=200).grid(row=2,column=0)
    matrix_B[0][0],matrix_B[0][1],matrix_B[0][2]=EntryB_get1()
    matrix_B[1][0],matrix_B[1][1],matrix_B[1][2]=EntryB_get2()
    matrix_B[2][0],matrix_B[2][1],matrix_B[2][2]=EntryB_get3()
    if np.linalg.det(matrix_B)==0:
        messagebox.showerror("Error","行列式為零，此矩陣不可逆")
    else:
        B_inv=np.linalg.inv(matrix_B)
        label_3=tk.Label(frame_3,text="逆矩陣:\n{}".format(B_inv),font=('Arial',16),bg="light blue")
        label_3.place(x=50,y=430)
def rank_B():
    frame_3=tk.Frame(win,bg="light blue",width=900,height=200).grid(row=2,column=0)
    matrix_B[0][0],matrix_B[0][1],matrix_B[0][2]=EntryB_get1()
    matrix_B[1][0],matrix_B[1][1],matrix_B[1][2]=EntryB_get2()
    matrix_B[2][0],matrix_B[2][1],matrix_B[2][2]=EntryB_get3()
    B_rank=np.linalg.matrix_rank(matrix_B)
    label_3=tk.Label(frame_3,text="秩: {}".format(B_rank),font=('Arial',16),bg="light blue")
    label_3.place(x=50,y=430)
def LUdecoposition_B():
    frame_3=tk.Frame(win,bg="light blue",width=900,height=200).grid(row=2,column=0)
    matrix_B[0][0],matrix_B[0][1],matrix_B[0][2]=EntryB_get1()
    matrix_B[1][0],matrix_B[1][1],matrix_B[1][2]=EntryB_get2()
    matrix_B[2][0],matrix_B[2][1],matrix_B[2][2]=EntryB_get3()
    if np.linalg.det(matrix_B)==0:
        messagebox.showerror("Error","行列式為零，此矩陣無法做LU分解")
    elif matrix_B[0][0]==0:
        messagebox.showwarning("Warning","請做PLU分解")
    else:
        U,L=LU_decomposition(matrix_B)
        label_3=tk.Label(frame_3,text="L:\n{}".format(L),font=('Arial',14),bg="light blue")
        label_3.place(x=50,y=430)
        label_4=tk.Label(frame_3,text="U:\n{}".format(U),font=('Arial',14),bg="light blue")
        label_4.place(x=470,y=430)
def PLUdecoposition_B():
    frame_3=tk.Frame(win,bg="light blue",width=900,height=200).grid(row=2,column=0)
    matrix_B[0][0],matrix_B[0][1],matrix_B[0][2]=EntryB_get1()
    matrix_B[1][0],matrix_B[1][1],matrix_B[1][2]=EntryB_get2()
    matrix_B[2][0],matrix_B[2][1],matrix_B[2][2]=EntryB_get3()
    P, L, U = lu(matrix_B)
    label_3=tk.Label(frame_3,text="P:\n{}".format(P),font=('Arial',12),bg="light blue")
    label_3.place(x=50,y=430)
    label_4=tk.Label(frame_3,text="L:\n{}".format(L),font=('Arial',12),bg="light blue")
    label_4.place(x=250,y=430)
    label_5=tk.Label(frame_3,text="U:\n{}".format(U),font=('Arial',12),bg="light blue")
    label_5.place(x=600,y=430)
def Choleskydecoposition_B():
    frame_3=tk.Frame(win,bg="light blue",width=900,height=200).grid(row=2,column=0)
    matrix_B[0][0],matrix_B[0][1],matrix_B[0][2]=EntryB_get1()
    matrix_B[1][0],matrix_B[1][1],matrix_B[1][2]=EntryB_get2()
    matrix_B[2][0],matrix_B[2][1],matrix_B[2][2]=EntryB_get3()
    if (matrix_B==matrix_B.T).all()==False:
        messagebox.showerror("Error","因為B≠B^T，無法做Cholesky分解")
    elif np.all(np.linalg.eigvals(matrix_B) > 0)==False:
        messagebox.showerror("Error","非正定矩陣，無法做Cholesky分解")
    else:
        L = np.linalg.cholesky(matrix_B)
        label_3=tk.Label(frame_3,text="Cholesky分解:\n{}".format(L),font=('Arial',14),bg="light blue")
        label_3.place(x=50,y=430)
        label_4=tk.Label(frame_3,text="\n{}".format(L.T),font=('Arial',14),bg="light blue")
        label_4.place(x=470,y=430)


# In[3]:


def add_A_B():
    frame_3=tk.Frame(win,bg="light blue",width=900,height=200).grid(row=2,column=0)
    matrix_A[0][0],matrix_A[0][1],matrix_A[0][2]=EntryA_get1()
    matrix_A[1][0],matrix_A[1][1],matrix_A[1][2]=EntryA_get2()
    matrix_A[2][0],matrix_A[2][1],matrix_A[2][2]=EntryA_get3()
    matrix_B[0][0],matrix_B[0][1],matrix_B[0][2]=EntryB_get1()
    matrix_B[1][0],matrix_B[1][1],matrix_B[1][2]=EntryB_get2()
    matrix_B[2][0],matrix_B[2][1],matrix_B[2][2]=EntryB_get3()
    add=matrix_A+matrix_B
    label_3=tk.Label(frame_3,text="A+B:\n{}".format(add),font=('Arial',16),bg="light blue")
    label_3.place(x=50,y=430)
def sub_A_B():
    frame_3=tk.Frame(win,bg="light blue",width=900,height=200).grid(row=2,column=0)
    matrix_A[0][0],matrix_A[0][1],matrix_A[0][2]=EntryA_get1()
    matrix_A[1][0],matrix_A[1][1],matrix_A[1][2]=EntryA_get2()
    matrix_A[2][0],matrix_A[2][1],matrix_A[2][2]=EntryA_get3()
    matrix_B[0][0],matrix_B[0][1],matrix_B[0][2]=EntryB_get1()
    matrix_B[1][0],matrix_B[1][1],matrix_B[1][2]=EntryB_get2()
    matrix_B[2][0],matrix_B[2][1],matrix_B[2][2]=EntryB_get3()
    sub=matrix_A-matrix_B
    label_3=tk.Label(frame_3,text="A-B:\n{}".format(sub),font=('Arial',16),bg="light blue")
    label_3.place(x=50,y=430)
def mul_A_B():
    frame_3=tk.Frame(win,bg="light blue",width=900,height=200).grid(row=2,column=0)
    matrix_A[0][0],matrix_A[0][1],matrix_A[0][2]=EntryA_get1()
    matrix_A[1][0],matrix_A[1][1],matrix_A[1][2]=EntryA_get2()
    matrix_A[2][0],matrix_A[2][1],matrix_A[2][2]=EntryA_get3()
    matrix_B[0][0],matrix_B[0][1],matrix_B[0][2]=EntryB_get1()
    matrix_B[1][0],matrix_B[1][1],matrix_B[1][2]=EntryB_get2()
    matrix_B[2][0],matrix_B[2][1],matrix_B[2][2]=EntryB_get3()
    mul=np.dot(matrix_A,matrix_B)
    label_3=tk.Label(frame_3,text="AxB:\n{}".format(mul),bg="light blue",font=('Arial',16))
    label_3.place(x=50,y=430)


# In[4]:


win=tk.Tk()                        # create window
win.title("3x3 Matrix Calculator")       # create title of window 
win.geometry('900x600')           # size of window
win.resizable(False,False)


frame_1=tk.Frame(win,bg="light blue",width=900,height=200).grid(row=0,column=0)
frame_2=tk.Frame(win,bg="light blue",width=900,height=200).grid(row=1,column=0)
frame_3=tk.Frame(win,bg="light blue",width=900,height=200).grid(row=2,column=0)


label_1=tk.Label(frame_1,text="Matrix A",bg="light blue",font=('Arial',16))
label_1.place(x=185, y=10)
label_2=tk.Label(frame_1,text="Matrix B",bg="light blue",font=('Arial',16))
label_2.place(x=635, y=10)

#A
entryA_1 = tk.Entry(frame_1,width=5,font=('Arial', 12))
entryA_1.place(x=100, y=50)
entryA_2 = tk.Entry(frame_1,width=5,font=('Arial', 12))
entryA_2.place(x=200, y=50)
entryA_3 = tk.Entry(frame_1,width=5,font=('Arial', 12))
entryA_3.place(x=300, y=50)
entryA_4 = tk.Entry(frame_1,width=5,font=('Arial', 12))
entryA_4.place(x=100, y=100)
entryA_5 = tk.Entry(frame_1,width=5,font=('Arial', 12))
entryA_5.place(x=200, y=100)
entryA_6 = tk.Entry(frame_1,width=5,font=('Arial', 12))
entryA_6.place(x=300, y=100)
entryA_7 = tk.Entry(frame_1,width=5,font=('Arial', 12))
entryA_7.place(x=100, y=150)
entryA_8 = tk.Entry(frame_1,width=5,font=('Arial', 12))
entryA_8.place(x=200, y=150)
entryA_9 = tk.Entry(frame_1,width=5,font=('Arial', 12))
entryA_9.place(x=300, y=150)

btnA_det=tk.Button(frame_2,text='行列式',width=10,font=('Arial', 12),bg="lemonchiffon",command=determinant_A) 
btnA_det.place(x=100,y=210)
btnA_transpose=tk.Button(frame_2,text='轉置矩陣',width=10,font=('Arial', 12),bg="lemonchiffon",command=transpose_A) 
btnA_transpose.place(x=250,y=210)
btnA_inv=tk.Button(frame_2,text='逆矩陣',width=10,font=('Arial', 12),bg="lemonchiffon",command=inverse_A) 
btnA_inv.place(x=100,y=260)
btnA_rank=tk.Button(frame_2,text='秩',width=10,font=('Arial', 12),bg="lemonchiffon",command=rank_A) 
btnA_rank.place(x=250,y=260)
btnA_LU=tk.Button(frame_2,text='LU分解',width=10,font=('Arial', 12),bg="lemonchiffon",command=LUdecoposition_A) 
btnA_LU.place(x=100,y=310)
btnA_PLU=tk.Button(frame_2,text='PLU分解',width=10,font=('Arial', 12),bg="lemonchiffon",command=PLUdecoposition_A) 
btnA_PLU.place(x=250,y=310)
btnA_Cholesky=tk.Button(frame_2,text='Cholesky分解',width=18,font=('Arial', 12),bg="lemonchiffon",command=Choleskydecoposition_A) 
btnA_Cholesky.place(x=140,y=360)


#B
entryB_1 = tk.Entry(frame_1,width=5,font=('Arial', 12))
entryB_1.place(x=550, y=50)
entryB_2 = tk.Entry(frame_1,width=5,font=('Arial', 12))
entryB_2.place(x=650, y=50)
entryB_3 = tk.Entry(frame_1,width=5,font=('Arial', 12))
entryB_3.place(x=750, y=50)
entryB_4 = tk.Entry(frame_1,width=5,font=('Arial', 12))
entryB_4.place(x=550, y=100)
entryB_5 = tk.Entry(frame_1,width=5,font=('Arial', 12))
entryB_5.place(x=650, y=100)
entryB_6 = tk.Entry(frame_1,width=5,font=('Arial', 12))
entryB_6.place(x=750, y=100)
entryB_7 = tk.Entry(frame_1,width=5,font=('Arial', 12))
entryB_7.place(x=550, y=150)
entryB_8 = tk.Entry(frame_1,width=5,font=('Arial', 12))
entryB_8.place(x=650, y=150)
entryB_9 = tk.Entry(frame_1,width=5,font=('Arial', 12))
entryB_9.place(x=750, y=150)

btnB_det=tk.Button(frame_2,text='行列式',width=10,font=('Arial', 12),bg="navajowhite",command=determinant_B) 
btnB_det.place(x=550,y=210)
btnB_transpose=tk.Button(frame_2,text='轉置矩陣',width=10,font=('Arial', 12),bg="navajowhite",command=transpose_B) 
btnB_transpose.place(x=700,y=210)
btnB_inv=tk.Button(frame_2,text='逆矩陣',width=10,font=('Arial', 12),bg="navajowhite",command=inverse_B) 
btnB_inv.place(x=550,y=260)
btnB_rank=tk.Button(frame_2,text='秩',width=10,font=('Arial', 12),bg="navajowhite",command=rank_B) 
btnB_rank.place(x=700,y=260)
btnB_LU=tk.Button(frame_2,text='LU分解',width=10,font=('Arial', 12),bg="navajowhite",command=LUdecoposition_B) 
btnB_LU.place(x=550,y=310)
btnB_PLU=tk.Button(frame_2,text='PLU分解',width=10,font=('Arial', 12),bg="navajowhite",command=PLUdecoposition_B) 
btnB_PLU.place(x=700,y=310)
btnB_Cholesky=tk.Button(frame_2,text='Cholesky分解',width=18,font=('Arial', 12),bg="navajowhite",command=Choleskydecoposition_B) 
btnB_Cholesky.place(x=590,y=360)

#A&B
btn_add=tk.Button(frame_2,text='A+B',width=8,font=('Arial', 12),bg="pink",command=add_A_B) 
btn_add.place(x=410,y=210)
btn_sub=tk.Button(frame_2,text='A-B',width=8,font=('Arial', 12),bg="pink",command=sub_A_B) 
btn_sub.place(x=410,y=285)
btn_mul=tk.Button(frame_2,text='AxB',width=8,font=('Arial', 12),bg="pink",command=mul_A_B) 
btn_mul.place(x=410,y=360)

win.mainloop()

