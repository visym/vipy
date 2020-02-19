import os
from vipy.util import remkdir, readjson, readyaml, findyaml, findvideo, filetail, findjson
from vipy.video import VideoCategory, Scene
from vipy.object import Track, Activity
from vipy.geometry import BoundingBox
import numpy as np
import warnings

class Mevadata_Public_01(object):
    def __init__(self, videodir, repodir):
        """mevadata-python-01:  http://mevadata.org"""
        self.videodir = videodir
        self.repodir = repodir

        assert os.path.exists(os.path.join(self.videodir, 'drop-01')), "Invalid input - videodir must contain the drop-01, drop-02 and drop-03 subdirectories.  See http://mevadata.org/#getting-data"
        assert os.path.exists(os.path.join(self.repodir, 'annotation')), "Invalid input - repodir must contain the clone of https://gitlab.kitware.com/meva/meva-data-repo"

    def __repr__(self):
        return str('<vipy.dataset.meva: videos="%s", annotations="%s">' % (self.videodir, self.repodir))


    def activities(self):
        """Return a list of activities"""
        return sorted(list(readjson(os.path.join(self.repodir, 'annotation', 'DIVA-phase-2', 'activity-index.json')).keys()))

    def activities_to_required_objects(self):
        """Return a dictionary of activity keys to set of required objects.  This is currently wrong."""
        warnings.warn('This mapping is currently wrong in the Kitware repository')
        d = readjson(os.path.join(self.repodir, 'annotation', 'DIVA-phase-2', 'activity-index.json'))
        return {a:set([x.replace('Construction_Vehicle', 'Vehicle') for x in d[a]['objectTypes']]) for a in self.activities()}        

    def _get_activities_yaml(self):
        return sorted([x for x in findyaml(self.repodir) if 'activities.yml' in x])

    def _get_geom_yaml(self):
        return sorted([x for x in findyaml(self.repodir) if 'geom.yml' in x])

    def _get_types_yaml(self):
        return sorted([x for x in findyaml(self.repodir) if 'types.yml' in x])

    def _get_fileindex_json(self):
        return sorted([x for x in findjson(self.repodir) if 'file-index.json' in x])

    def _get_activities_json(self):
        return sorted([x for x in findjson(self.repodir) if 'activities.json' in x])
        
    def _get_videos(self):
        return sorted([x for x in findvideo(self.videodir)])

    def KF1_examples(self):
        """MEVA annotation format: https://gitlab.kitware.com/meva/meva-data-repo/blob/master/documents/MEVA_Annotation_JSON.pdf
        """
        d_videoname_to_path = {filetail(f):f for f in self._get_videos()}
        vidlist = []
        for (activities_jsonfile, fileindex_jsonfile) in zip(self._get_activities_json(), self._get_fileindex_json()):
            assert activities_jsonfile.split('.')[:-2] == fileindex_jsonfile.split('.')[:-2], "Unmatched activity and file-index json file"
            if 'KF1-examples' not in activities_jsonfile or 'KF1-examples' not in fileindex_jsonfile:
                continue
            activities = readjson(activities_jsonfile)
            fileindex = readjson(fileindex_jsonfile)
            assert len(fileindex) == 1, "Fileindex contains more than one video"
            
            # Create Scene() object
            videoname = tuple(fileindex.keys())[0]            
            assert 'selected' in fileindex[videoname] and 1 in fileindex[videoname]['selected'].values() and 0 in fileindex[videoname]['selected'].values(), "Invalid fileindex '%s'" % fileindex_jsonfile
            startframe = sorted([int(x) for x in fileindex[videoname]['selected'].keys()])[0]
            endframe = sorted([int(x) for x in fileindex[videoname]['selected'].keys()])[1]
            vid = Scene(filename=d_videoname_to_path[videoname], framerate=fileindex[videoname]['framerate'], startframe=startframe, endframe=endframe)

            # Add activities
            d = readjson(activities_jsonfile)
            for a in d['activities']:
                trackids = []
                for obj in a['objects']:
                    keyframes = sorted([int(k) for (k,x) in obj['localization'][videoname].items() if len(x)>0])
                    boxes = [BoundingBox(xmin=bb['x'], ymin=bb['y'], width=bb['w'], height=bb['h']) for bb in [obj['localization'][videoname][str(k)]['boundingBox'] for k in keyframes]]
                    t = Track(keyframes=keyframes, boxes=boxes, category=obj['objectType'], attributes=obj['objectID'], boundary='strict')
                    trackids.append(vid.add(t))

                startframe = sorted([int(x) for x in a['localization'][videoname].keys()])[0]
                endframe = sorted([int(x) for x in a['localization'][videoname].keys()])[1]
                vid.add(Activity(category=a['activity'], startframe=startframe, endframe=endframe, objectids=trackids, attributes={'activityID':a['activityID']}))                
            vidlist.append(vid)
        return vidlist

    def meva_annotations(self):
        """Kwiver packet format: https://gitlab.kitware.com/meva/meva-data-repo/blob/master/documents/KPF-specification-v4.pdf
        """

        d_videoname_to_path = {filebase(f):f for f in self._get_videos()}

        vidlist = []
        for (act_yamlfile, geom_yamlfile, types_yamlfile) in zip(self._get_activities_yaml(), self._get_geom_yaml(), self._get_types_yaml()):
            assert act_yamlfile.split('.')[:-2] == geom_yamlfile.split('.')[:-2], "Unmatched activity and geom yaml file"
            if 'meva-annotations' not in act_yamlfile:
                continue

            videoname = readyaml(types_yamlfile)[0]['meta']
            vid = Scene(filename=d_videoname_to_path[videoname], framerate=None)
            
            d_id1_to_category = {}
            for t in readyaml(types_yamlfile):
                if 'types' in t:
                    d_id1_to_category[t['types']['id1']] = list(t['types']['cset3'].keys())[0]

            d_trackid_to_track = {}
            for v in readyaml(geom_yamlfile):
                if 'geom' in v:
                    keyframe = v['geom']['id1']['ts0']
                    bb = [int(x) for x in v['geom']['id1']['g0'].split(' ')]
                    bbox = BoundingBox(xmin=bb[0], ymin=bb[1], xmax=bb[2], ymax=bb[3])
                    if v['geom']['id1'] not in d_trackid_to_track:
                        d_trackid_to_track[v['geom']['id1']] = Track(category=d_id1_to_category[v['geom']['id1']], framerate=None)
                    d_trackid_to_track[v['geom']['id1']].add(keyframe=keyframe, box=bbox)
                
            for (k,v) in d_trackid_to_track.items():
                vid.add(v)

            for v in readyaml(act_yamlfile):
                if 'act' in v:
                    if 'act2' in v['act']:
                        category = list(v['act']['act2'].keys())[0]
                    elif 'act3' in v['act']:
                        category = list(v['act']['act3'].keys())[0]
                    else:
                        raise ValueError('Invalid activity YAML - act2 or act3 must be specified')
                    startframe = v['act']['timespan']['tsr0'][0]
                    endframe = v['act']['timespan']['tsr0'][1]
                    actorid = [x['id1'] for x in v['act']['actors']]
                    trackids = [d_trackid_to_track[x] for x in actorid]

                    vid.add(Activity(category=category, startframe=startframe, endframe=endframe, objectids=trackids))
            
            vidlist.append(vid)
        return vidlist
