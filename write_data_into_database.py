import mysql.connector

mydb = mysql.connector.connect(
  host="localhost",
  user="byt",
  password="1231",
  database="interaction"
)
while True:
    flag = input("flag = ")
    mycursor = mydb.cursor()

    sql = "INSERT INTO loading (id, flag) VALUES (%s, %s)"
    val = (None,flag)
    mycursor.execute(sql, val)
    mydb.commit()
    print("1 record inserted, ID:", mycursor.lastrowid)