from vipy.util import to_iterable, truncate_string
from itertools import zip_longest


try:
    import ujson as json  # faster
except ImportError:
    import json

            
class Tag():
    """vipy.label.Tag class

    A Tag label is one or more strings with optional confidences that provide ground truth or estimated keywords/tags/labels/classes

    ```python
    tag = vipy.label.Tag('person')
    tag = vipy.label.Tag('person', confidence=0.5)
    tag = vipy.label.Tag( (('person', 0.5), ('other_person', 0.1)) )
    ```

    The category of a tag is the first tag
    The confidence of a tag is the confidence of the first tag
    
    """
    def __init__(self, tags=None, category=None, confidence=None):
        self._tags = []
        
        if category is not None:
            self.category(category, confidence)
        if tags is not None:
            for t in to_iterable(tags):                
                self.tag(t,None) if not isinstance(t, (tuple,list)) else self.tag(*t)

    def __repr__(self):
        fields  = ['category=%s' % truncate_string(str(self.category()), 40)] if self.has_category() else []
        fields += ['confidence=%1.3f' % self.confidence()] if self.has_category() and self.confidence() is not None else []        
        fields += ['tags=%s' % truncate_string(str(self.tags()), 40)] if self.has_tags() and not self.has_category() else []
        return str('<vipy.label.Tag: %s>' % (', '.join(fields)))

    def __or__(self, other):
        self._tags += other._tags if hasattr(other, '_tags') else []
        return self

    def dict(self):
        return {'tag': self._tags}
    
    def json(self):
        return json.dumps(self.dict()) 
        
    @classmethod
    def from_json(obj, s):
        return obj(tags=json.loads(s)['tag'])

    def clear_tags(self):
        self._tags = []
        return self

    def tag(self, tag, confidence=None):
        assert isinstance(confidence, (tuple, list, float, int, type(None))), "invalid type '%s'" % (type(confidence))                
        for (t,c) in zip_longest(to_iterable(tag), to_iterable(confidence)):
            self._tags.append( t if c is None else (t, float(c)) )
        return self

    def tags(self):
        return tuple(t[0] if isinstance(t, (tuple,list)) else t for t in self._tags)
    
    def has_tag(self, t):
        return t in self.tags()

    def num_tags(self):
        return len(self.tags())
    
    def has_tags(self):
        return self.num_tags() > 0
    
    def soft_tags(self):
        return tuple(t if isinstance(t, (tuple,list)) else (t,None) for t in self._tags)        

    def has_soft_tags(self):
        return all(t[1] is not None for t in self.soft_tags())
    
    def first_tag(self):
        return self.tags()[0]
    
    def has_category(self):
        return self.num_tags() == 1

    def clear_category(self):
        return self.clear_tags()

    def no_category(self):
        return self.clear_tags()

    def categoryif(self, old, new):
        if self.category() == old:
            self.category(new)
        return self

    def new_category(self, category):
        self.category(category)
        return self
    
    def category(self, category=None, confidence=None):
        if category is not None or confidence is not None:
            assert isinstance(confidence, (float, int, type(None))), "invalid type '%s'" % (type(confidence))                                        
            (tag, conf) = (self.category() if category is None else category, self.confidence() if confidence is None else float(confidence))
            self._tags = [(tag, conf)] + self._tags[1:]  # category is always the first tag
            return self
        return self.first_tag() if self.has_tags() else None

    def label(self):
        """synonym for category"""
        return self.category()
    
    def confidence(self, conf=None):
        return (self._tags[0][1] if isinstance(self._tags[0], (tuple,list)) else None) if self.has_tags() else None

    def has_confidence(self):
        return self.confidence() is not None
    
    def is_unlabeled(self):
        return self.num_tags() == 0

    
class Category(Tag):
    """vipy.label.Category class

    A category label is a tag with at most one tag        
    """
    
    def __init__(self, category=None, confidence=None):
        super().__init__(category=category, confidence=confidence)
    
    def tag(self, tag, confidence=None):
        assert self.num_tags() == 0, "vipy.label.Category has at most one tag"
        return super().tag(tag, confidence)

    def __repr__(self):
        fields  = ['category=%s' % truncate_string(str(self.category()), 40)] if self.has_category() else []
        fields += ['confidence=%1.3f' % self.confidence()] if self.has_category() and self.confidence() is not None else []        
        return str('<vipy.label.Category: %s>' % (', '.join(fields)))
    
    
class Caption():
    """vipy.label.Caption class

    A caption is tag label is one or more natural language strings with optional confidences that provide ground truth or estimated descriptions

    ```python
    caption = vipy.label.Caption('An image containing a person')
    caption = vipy.label.Caption((('An image containing a person', 0.99), ('An image containing two people', 0.1)))
    ```
    
    """    
    def __init__(self, captions=None):
        self._captions = []

        if captions is not None:
            for c in to_iterable(captions):
                self.caption(c,None) if not isinstance(c, (tuple,list)) else self.caption(*c)                
            
    def __repr__(self):
        fields = ['captions=%s' % truncate_string(str(self.captions()), 40)] if self.has_captions() else []
        return str('<vipy.label.Caption: %s>' % (', '.join(fields)))

    def __or__(self, other):
        self._captions += other._captions if hasattr(other, '_captions') else []
        return self

    def dict(self):
        return {'captions': self._captions}
    
    def json(self):
        return json.dumps(self.dict()) 

        
    @classmethod
    def from_json(obj, s):
        return obj(captions=json.loads(s)['captions'])

    def clear_captions(self):
        self._captions = []
        return self

    def num_captions(self):
        return len(self.captions())
    
    def has_captions(self):
        return self.num_captions() > 0

    def captions(self):
        return tuple(c if not isinstance(c, (tuple,list)) else c[0] for c in self._captions)

    def caption(self, caption, confidence=None):
        assert isinstance(confidence, (tuple, list, float, int, type(None))), "invalid type '%s'" % (type(confidence))        
        for (t,c) in zip_longest(to_iterable(caption), to_iterable(confidence)):
            assert c is None or isinstance(c, (int, float))
            self._captions.append( t if c is None else (t, float(c)) )
        return self
    
    def is_unlabeled(self):
        return self.num_captions() == 0


class Label(Tag, Caption):
    """vipy.label.Label class

    A label is a `vipy.label.Tag` and `vipy.label.Caption`.
    
    """
    
    def __init__(self, tags=None, captions=None, category=None):
        Tag.__init__(self, tags=tags, category=category)
        Caption.__init__(self, captions=captions)
        
    def __repr__(self):
        fields  = ['category=%s' % truncate_string(str(self.category()), 40)] if self.has_category() else []
        fields += ['confidence=%1.3f' % self.category_confidence()] if self.has_category() and self.category_confidence() is not None else []        
        fields += ['tags=%s' % truncate_string(str(self.tags()), 40)] if self.has_tags() and not self.has_category() else []
        fields += ['captions=%s' % truncate_string(str(self.captions()), 40)] if self.has_captions() else []
        return str('<vipy.label.Label: %s>' % (', '.join(fields)))

    def __or__(self, other):
        Tag.__or__(self, other)
        Caption.__or__(self, other)
        return self

    def dict(self):
        return Tag.dict(self) | Caption.dict(self)

    def json(self):
        return json.dumps(self.dict()) 
        
    @classmethod
    def from_json(obj, s):
        d = json.loads(s)
        return obj(captions=d['captions'], tags=d['tags'])
    
    def is_unlabeled(self):
        return Tag.is_unlabeled(self) and Caption.is_unlabeled(self)
    

    
