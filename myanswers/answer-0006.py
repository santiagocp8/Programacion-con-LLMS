def resumir_interferencias(df):
    return (
        df.groupby('nivel_edificio')[['id_clash']]
        .count()
        .rename(columns={'id_clash': 'total_conflictos'})
    )
