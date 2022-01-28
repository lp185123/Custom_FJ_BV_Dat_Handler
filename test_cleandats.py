import DatScraper_tool2


imageExtractor = DatScraper_tool2.ImageExtractor(r'C:\Working\FindIMage_In_Dat\Input\0001.dat')
#
noteIds = [1,2,3]
# 
imageExtractor.clean(noteIds)