def detectar_libros_populares(df, factor):
    mean_borrow = df["borrow_count"].mean()
    return df[df["borrow_count"] > mean_borrow * factor]
