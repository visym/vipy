import os
from vipy.util import remkdir, readjson, readyaml, findyaml, findvideo, filetail, findjson, filebase, readlist, groupbyasdict
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
        """Parse KF1-examples annotations from 'meva-data-repo/annotation/DIVA-phase-2/MEVA/KF1-examples' to list of vipy.video.Scene()

        MEVA annotation format: https://gitlab.kitware.com/meva/meva-data-repo/blob/master/documents/MEVA_Annotation_JSON.pdf

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

    def MEVA(self, verbose=True, stride=1):
        """Parse MEVA annotations from 'meva-data-repo/annotation/DIVA-phase-2/MEVA/meva-annotations/' into vipy.video.Scene()
        
        Kwiver packet format: https://gitlab.kitware.com/meva/meva-data-repo/blob/master/documents/KPF-specification-v4.pdf

        """

        d_videoname_to_path = {filebase(f):f for f in self._get_videos()}
        if verbose:
            num_yamlfiles = len(self._get_activities_yaml())
            print('[vipy.dataset.meva]: Parsing %d YAML files, this will take a while because pure python YAML loader is slow...' % num_yamlfiles)

        vidlist = []
        for (k_fileindex, (act_yamlfile, geom_yamlfile, types_yamlfile)) in enumerate(zip(self._get_activities_yaml(), self._get_geom_yaml(), self._get_types_yaml())):
            if verbose:
                print('[vipy.dataset.meva][%d/%d]: Parsing "%s"' % (k_fileindex+1, num_yamlfiles, act_yamlfile))
            assert act_yamlfile.split('.')[:-2] == geom_yamlfile.split('.')[:-2], "Unmatched activity and geom yaml file"
            if 'meva-annotations' not in act_yamlfile:
                continue

            # This is surprisingly slow...
            types_yaml = readyaml(types_yamlfile)
            geom_yaml = readyaml(geom_yamlfile)
            act_yaml = readyaml(act_yamlfile)

            assert len(set([types_yaml[0]['meta'], geom_yaml[0]['meta'], act_yaml[0]['meta']]))==1, "Mismatched video name for '%s'" % act_yamlfile
            videoname = act_yaml[0]['meta']
            if videoname not in d_videoname_to_path:
                warnings.warn('Invalid video "%s" in "%s" - Ignoring' % (videoname, filebase(act_yamlfile)))
                continue

            # Parse
            framerate = 30.0  # All videos are universally 30Hz (from Roddy)
            vid = Scene(filename=d_videoname_to_path[videoname], framerate=framerate)

            d_id1_to_category = {}
            for t in types_yaml:
                if 'types' in t:
                    d_id1_to_category[t['types']['id1']] = list(t['types']['cset3'].keys())[0]

            d_id1_to_track = {}
            d_geom_yaml = groupbyasdict([x['geom'] for x in geom_yaml if 'geom' in x], lambda v: v['id1'])
            assert stride >= 1, "Invalid stride"
            for (id1, geom_yaml) in d_geom_yaml.items():
                for (k_geom, v) in enumerate(geom_yaml):
                    if stride > 1 and k_geom > 0 and (k_geom < (len(geom_yaml)-stride)) and (k_geom % stride == 0):
                        continue  # Use vipy track interpolation to speed up parsing
                    keyframe = v['ts0']
                    bb = [int(x) for x in v['g0'].split(' ')]
                    bbox = BoundingBox(xmin=bb[0], ymin=bb[1], xmax=bb[2], ymax=bb[3])
                    if not bbox.isvalid():
                        warnings.warn('Invalid bounding box: id1=%s, bbox="%s", file="%s" - Ignoring' % (str(v['id1']), str(bbox), filetail(geom_yamlfile)))
                    elif v['id1'] not in d_id1_to_track:
                        d_id1_to_track[v['id1']] = Track(category=d_id1_to_category[v['id1']], framerate=framerate, keyframes=[keyframe], boxes=[bbox])
                    else:
                        d_id1_to_track[v['id1']].add(keyframe=keyframe, box=bbox)
                
            for (k,v) in d_id1_to_track.items():
                vid.add(v)

            for v in act_yaml:
                if 'act' in v:
                    if 'act2' in v['act']:
                        category = list(v['act']['act2'].keys())[0]
                    elif 'act3' in v['act']:
                        category = list(v['act']['act3'].keys())[0]
                    else:
                        raise ValueError('Invalid activity YAML - act2 or act3 must be specified')
                    assert len(v['act']['timespan']) == 1, "Multi-span activities not parsed"
                    startframe = int(v['act']['timespan'][0]['tsr0'][0])
                    endframe = int(v['act']['timespan'][0]['tsr0'][1])
                    actorid = [x['id1'] for x in v['act']['actors']]
                    trackids = [d_id1_to_track[x].id() for x in actorid]

                    vid.add(Activity(category=category, startframe=startframe, endframe=endframe, objectids=trackids, framerate=framerate, attributes={'src_status':v['act']['src_status']}))
            
            vidlist.append(vid)
        return vidlist
