import csv

class ImageStream(object):
    def __init__(self, csvfile, parser, delimiter=' ', quotechar='|', rowstart=1):        
        self._csvfile = csvfile
        self._parser = parser
        self._delimiter = delimiter
        self._quotechar = quotechar
        self._rowstart = rowstart
            
    def __iter__(self):
        self._f_csv = open(self._csvfile, 'rb')
        self._reader = csv.reader(self._f_csv, delimiter=self._delimiter, quotechar=self._quotechar)  # reopen
        if self._rowstart > 1:
            for k in range(1, self._rowstart):
                self._reader.next()
        return self

    def __getitem__(self, item):
        # Inefficient method for random stream access
        f = open(self._csvfile, 'r')
        for i in range(max(item,1)):
            line = f.readline()
            if len(line) == 0:
                raise IndexError('Invalid index "%d"' % item)  # end of file
        f.close()
        return self._parser(line)

    def next(self):
        row = self._reader.next()
        return self._parser(row)
    

class VideoStream(ImageStream):
    _f_csv = None
    firstframe = None
    
    def __init__(self, csvfile, keyby, parser, delimiter=' ', quotechar='|', rowstart=1):        
        self._csvfile = csvfile
        self._parser = parser
        self._delimiter = delimiter
        self._quotechar = quotechar
        self._rowstart = rowstart
        self._keyby = keyby

    def __iter__(self):
        self._f_csv = open(self._csvfile, 'rb')
        self._reader = csv.reader(self._f_csv, delimiter=self._delimiter, quotechar=self._quotechar)  # reopen
        if self._rowstart > 1:
            for k in range(1, self._rowstart):
                self._reader.next()
        return self
        
    def __getitem__(self, item):
        # Inefficient method for random stream access
        self._rowstart = item
        return self._parser( [self.__iter__().next()] )   # return singleton list for consistency with next()
        
    def next(self):
        """Read at least one frame in row order until keyby field changes, then parse the list of rows into a video object"""
        is_finalframe = False
        firstframe = self.firstframe if self.firstframe is not None else self._reader.next()
        firstkey = self._keyby(firstframe)
        frames = [firstframe]
        while not is_finalframe:
            row = self._reader.next()
            framekey = self._keyby(row)
            if framekey == firstkey:
                frames.append(row)
            else:
                is_finalframe = True
            self.firstframe = row            
        return self._parser(frames)
    
