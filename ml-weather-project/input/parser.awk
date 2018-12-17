BEGIN {
  FS=","
  OFS=","
  RS="\n"
  ORS="\n"
}

{
  if (NR == 1) {
    print($0)

  } else {
  n_matches = match($8, /-([0-9][0-9])-/, arr)
  month = int(arr[1])

  jan = int(month == 1)
  feb = int(month == 2)
  mar = int(month == 3)
  apr = int(month == 4)
  may = int(month == 5)
  jun = int(month == 6)
  jul = int(month == 7)
  aug = int(month == 8)
  sep = int(month == 9)
  oct = int(month == 10)
  nov = int(month == 11)
  dec = int(month == 12)

 print($0, jan, feb, mar, apr, may, jun, jul, aug, sep, oct, nov, dec)
}
}
