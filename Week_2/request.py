import wrds
def request():
#Login
    db = wrds.Connection(wrds_username="razevedo")
###Main
#Create a stock: AAPL#7, MSFT#8048, GE#20792, PG#21446, GS#35048
    aapl = Stock("AAPL", permco = 7)
    msft = Stock("MSFT", permco = 8048)
    ge = Stock("GE", permco = 20792)
    pg = Stock("PG", permco = 21446)
    gs = Stock("GS", permco = 35048)

#Bundle together in a unique array
    df = download_bundle_stock([aapl, msft, ge, pg, gs], initial_date= '2001/01/01', final_date= '2024/12/31')
    print(df.head(5))

#Save as csv file
    df.to_csv('data.csv', ';', index = True)