f=open('./data/movieReviews1000.txt',"r")
#contents is a list with each elements as a line of the text
text=f.readlines()

#removing End Labels
# label=[]
# for i in range(len(text)):
#     label.append((int)(text[i][len(text[i])-2]))
#     text[i]=text[i][:(len(text[i])-2)]
    
labels = []
reviews = []
for line in text:
    review, label = line.split('\t')
    labels.append(int(label.strip()))
    reviews.append(review.strip())

