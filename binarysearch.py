
Array1=[-1,3,8,2,9,5,30,40,50]
Array2=[4,1,2,10,5,20,30,40,50]

#brute force solution
BestMatch=0
NumberToMatch=24
for Index,Elem in enumerate(Array1):
  for Index2,Elem2 in enumerate(Array2):
    Result=abs(NumberToMatch-(Elem+Elem2))
    if Index + Index2 ==0:
      BestMatch=Result
    if Result<BestMatch:
      BestMatch=Result
      print(f"sum of array 1 {Elem} and array 2 {Elem2}")
      print(f"Best product closest to {NumberToMatch} is {Result}")



#binary search solution
#sort arrays
Array1.sort()
Array2.sort()

def BinaryClosestSearch(Sequence,item,BestMatch):
  begin_index=0
  end_index=len(Sequence)-1
  BestMatch_Index=None
  Evaluation=None
  while True:#begin_index<end_index:
    midpoint=begin_index+((end_index-begin_index)//2)
    print(f"Midpoint {Sequence[midpoint]}")
    #evaluate
    Evaluation=abs(24-(Sequence[midpoint]+item))
    if BestMatch is None:
        BestMatch=(Evaluation,midpoint,Sequence[midpoint])
    if Evaluation<BestMatch[0]:
      BestMatch=(Evaluation,midpoint,Sequence[midpoint])


    if Sequence[midpoint]+item==24:
        BestMatch_Index=(0,midpoint)
        return BestMatch_Index
    elif (Sequence[midpoint]+item)<24:
        begin_index=midpoint+1
    else:
        end_index=midpoint-1
    #update if a better result
    if end_index<begin_index or begin_index>end_index:
        break

  return BestMatch




for Elem in Array1:
    BestMatch=BinaryClosestSearch(Array2,Elem,None)
    print(f"best match {BestMatch} for number {Elem} looking in array {Array2} ")
      