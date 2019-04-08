# BOOLEAN MASKIBG - CLASS OBJECT
# BM - Boolean Masking
class BM:
    def  __init__(self,df):
        self.df= df
    
    def select( self, column , operator , value):
        if operator ==  "equals":
            return BM(self.df[self.df[column]==value])
        elif operator == "contains":
            return BM(self.df[[(value.lower() in str(title).lower()) for title in self.df[column]]])
        elif operator ==  "gt":
            return BM(self.df[self.df[column] > value])
        elif operator ==  "ge":
            return BM(self.df[self.df[column] >= value])
        elif operator ==  "lt":
            return BM(self.df[self.df[column] <= value])
        elif operator ==  "le":
            return BM(self.df[self.df[column] <= value])
        elif operator ==  "ne":
            return BM(self.df[self.df[column] != value])

