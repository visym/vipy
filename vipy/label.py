import json
from vipy.util import to_iterable, truncate_string

            
class Tag():
    """vipy.label.Tag class

    A Tag label is one or more strings with optional confidences that provide ground truth or estimated keywords/tags/labels/classes
    """
    def __init__(self, tags=None, category=None, confidence=None):
        if not hasattr(self, 'attributes'):
            self.attributes = {}
        if category is not None:
            self.category(category, confidence)
        if tags is not None:
            for t in to_iterable(tags):
                self.tag( *((t['tag'], t['confidence']) if isinstance(t, dict) else (t, None)) )        

    def __repr__(self):
        fields  = ['category=%s' % truncate_string(str(self.category()), 40)] if self.has_category() else []
        fields += ['confidence=%1.3f' % self.category_confidence()] if self.has_category() and self.confidence() is not None else []        
        fields += ['tags=%s' % truncate_string(str(self.tags()), 40)] if self.has_tags() and not self.has_category() else []
        return str('<vipy.label.Tag: %s>' % (', '.join(fields)))

    def __or__(self, other):
        self.attributes['tags'] = ((self.attributes['tags'] if self.has_tags() else []) +
                                   (other.attributes['tags'] if hasattr(other, 'attributes') and 'tags' in other.attributes else []))
        return self

    def dict(self):
        return {'vipy.label.Tag': self.attributes['tags']}
    
    def json(self):
        return json.dumps(self.dict()) 
        
    @classmethod
    def from_json(obj, s):
        return obj(tags=json.loads(s)['vipy.label.Tag'])

    def clear_tags(self):
        self.attributes['tags'] = []
        return self

    def tag(self, tag, confidence=None):
        if not self.has_tags():
            self.attributes['tags'] = []
        self.attributes['tags'].append( {'tag':tag, 'confidence':confidence}  )
        return self

    def tags(self):
        return tuple(t['tag'] for t in self.attributes['tags']) if self.has_tags() else ()
    
    def has_tag(self, t):
        return t in self.tags()

    def num_tags(self):
        return len(self.tags())
    
    def has_tags(self):
        return 'tags' in self.attributes
    
    def soft_tags(self):
        return tuple((t['tag'], t['confidence']) for t in self.attributes['tags']) if self.has_tags() else ()

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

    def new_category(self, category, confidence=None):
        self.category(category, confidence)
        return self
    
    def category(self, category=None, confidence=None):
        if category is not None or confidence is not None:
            self.clear_tags().tag(category if category is not None else self.category(), confidence)
            return self
        return self.first_tag() if self.has_tags() else None

    def label(self):
        """synonym for category"""
        return self.category()
    
    def confidence(self):
        return self.attributes['tags'][0]['confidence'] if self.has_tags() else None

    def has_confidence(self):
        return self.confidence() is not None
    
    def is_unlabeled(self):
        return self.num_tags() == 0

    
class Category(Tag):
    """vipy.label.Category class

    A Category is a `vipy.label.Tag` with at most one tag
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
    def __init__(self, captions=None):
        if not hasattr(self, 'attributes'):
            self.attributes = {}
        if captions is not None:
            for c in to_iterable(captions):
                self.caption( *((c['caption'], c['confidence']) if isinstance(c, dict) else (c, None)) )                        
                self.caption(c)
            
    def __repr__(self):
        fields = ['captions=%s' % truncate_string(str(self.captions()), 40)] if self.has_captions() else []
        return str('<vipy.label.Caption: %s>' % (', '.join(fields)))

    def __or__(self, other):
        self.attributes['captions'] = ((self.attributes['captions'] if self.has_captions() else []) +
                                       (other.attributes['captions'] if hasattr(other, 'attributes') and 'captions' in other.attributes else []))
        return self

    def dict(self):
        return {'vipy.label.Caption': self.attributes['captions']}
    
    def json(self):
        return json.dumps(self.dict()) 

        
    @classmethod
    def from_json(obj, s):
        return obj(captions=json.loads(s)['vipy.label.Caption'])

    def clear_captions(self):
        self.attributes['captions'] = []
        return self

    def num_captions(self):
        return len(self.captions())
    
    def has_captions(self):
        return 'captions' in self.attributes and len(self.attributes['captions'])>0

    def captions(self):
        return tuple(c['caption'] for c in self.attributes['captions']) if self.has_captions() else ()
    
    def caption(self, caption, confidence):
        if not self.has_captions():
            self.attributes['captions'] = []
        self.attributes['captions'].append( {'caption':caption, 'confidence':confidence} )
        return self
            
    def is_unlabeled(self):
        return self.num_captions() == 0


class Label(Tag, Caption):
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
        return {'vipy.label.Label': Tag.dict(self) | Caption.dict(self)}

    def json(self):
        return json.dumps(self.dict()) 
        
    @classmethod
    def from_json(obj, s):
        d = json.loads(s)
        return obj(captions=d['vipy.label.Label']['vipy.label.Caption'],
                   tags=d['vipy.label.Label']['vipy.label.Tag'])
    
    def is_unlabeled(self):
        return Tag.is_unlabeled(self) and Caption.is_unlabeled(self)
    

    
