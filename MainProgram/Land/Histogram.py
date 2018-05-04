 for sp in spread1.index:
    plt.rcParams['figure.figsize'] = [14, 8]  
    print(sp)
    df[sp].hist() 
    plt.title('Distribution of maxhumidity_1')  
    plt.xlabel(sp)  
    plt.show()