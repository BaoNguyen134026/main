import mysql.connector

mydb = mysql.connector.connect(
  host="localhost",
  user="byt",
  password="1231",
  database="interaction"
)
while True:
    g = input("put your g:")
    h = input("put your h:")
    v1 = input("put your v1:")
    v2 = input("put your v2:")
    v3 = input("put your v3:")

    mycursor = mydb.cursor()
    sql = "INSERT INTO size (id, g,h,v1,v2,v3) VALUES (%s, %s,%s, %s,%s, %s)"
    val = (None,g,h,v1,v2,v3)
    mycursor.execute(sql, val)
    mydb.commit()
    print("1 record inserted, ID:", mycursor.lastrowid)