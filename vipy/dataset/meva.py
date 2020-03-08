import os
from vipy.util import remkdir, readjson, readyaml, findyaml, findvideo, filetail, findjson, filebase, readlist, groupbyasdict, save, flatlist, isstring, tempdir
from vipy.video import VideoCategory, Scene
from vipy.object import Track, Activity
from vipy.geometry import BoundingBox
from vipy.batch import Batch
from vipy.show import colorlist
import vipy.metrics
import numpy as np
import warnings
import shutil


class Mevadata_Public_01(object):
    def __init__(self, videodir, repodir):
        """mevadata-python-01:  http://mevadata.org"""
        self.videodir = videodir
        self.repodir = repodir

        assert os.path.exists(os.path.join(self.videodir, 'drop-01')), "Invalid input - videodir must contain the drop-01, drop-02 and drop-03 subdirectories.  See http://mevadata.org/#getting-data"
        assert os.path.exists(os.path.join(self.repodir, 'annotation')), "Invalid input - repodir must contain the clone of https://gitlab.kitware.com/meva/meva-data-repo"

        self._d_category_to_shortlabel = {'person_abandons_package':'Abandoning',
                                          'person_closes_facility_door':'Closing',
                                          'person_closes_trunk':'Closing trunk',
                                          'person_closes_vehicle_door':'Closing door',
                                          'person_embraces_person':'Hugging',
                                          'person_enters_scene_through_structure':'Entering',
                                          'person_enters_vehicle':'Entering',
                                          'person_exits_scene_through_structure':'Exiting',
                                          'person_exits_vehicle':'Exiting',
                                          'hand_interacts_with_person':'Using hand',
                                          'person_carries_heavy_object':'Carrying',
                                          'person_interacts_with_laptop':'Using laptop',
                                          'person_loads_vehicle':'Loading',
                                          'person_transfers_object':'Transferring',
                                          'person_opens_facility_door':'Opening door',
                                          'person_opens_trunk':'Opening trunk',
                                          'person_opens_vehicle_door':'Opening door',
                                          'person_talks_to_person':'Talking',
                                          'person_picks_up_object':'Picking up',
                                          'person_purchases':'Purchasing',
                                          'person_reads_document':'Reading',
                                          'person_rides_bicycle':'Riding',
                                          'person_puts_down_object':'Putting down',
                                          'person_sits_down':'Sitting',
                                          'person_stands_up':'Standing',
                                          'person_talks_on_phone':'Talking',
                                          'person_texts_on_phone':'Texting',
                                          'person_steals_object':'Stealing',
                                          'person_unloads_vehicle':'Unloading',
                                          'vehicle_drops_off_person':'Dropping off',
                                          'vehicle_picks_up_person':'Picking up',
                                          'vehicle_reverses':'Reversing',
                                          'vehicle_starts':'Starting',
                                          'vehicle_stops':'Stopping',
                                          'vehicle_turns_left':'Turning left',
                                          'vehicle_turns_right':'Turning right',
                                          'vehicle_makes_u_turn':'Turning around'}
        self._d_category_to_shortlabel = {k:v.lower() for (k,v) in self._d_category_to_shortlabel.items()}
        
        self._d_oldcategory_to_shortlabel = {'Closing_Trunk':'Closing',
                                             'Open_Trunk':'Opening',
                                             'Riding':'Riding',
                                             'Talking':'Talking',
                                             'person_talks_to_person':'Talking',                                          
                                             'Transport_HeavyCarry':'Carrying',
                                             'Unloading':'Unloading',
                                             'abandon_package':'Abandoning',
                                             'hand_interaction':'Using Hand',
                                             'object_transfer':'Transferring',
                                             'person_closes_facility_door':'Closing',
                                             'person_closes_vehicle_door':'Closing',
                                             'person_enters_through_structure':'Entering',
                                             'person_enters_vehicle':'Entering',
                                             'person_exits_through_structure':'Exiting',                                          
                                             'person_exits_vehicle':'Exiting',
                                             'person_laptop_interaction':'Interacting',
                                             'person_loads_vehicle':'Loading',
                                             'person_opens_facility_door':'Opening',
                                             'person_opens_vehicle_door':'Opening',
                                             'person_person_embrace':'Hugging',
                                             'person_picks_up_object':'Picking up',
                                             'person_purchasing':'Purchasing',
                                             'person_reading_document':'Reading',
                                             'person_sets_down_object':'Setting down',
                                             'person_sitting_down':'Sitting',
                                             'person_standing_up':'Standing',
                                             'person_stands_up':'Standing',                                          
                                             'specialized_talking_phone':'Talking',
                                             'specialized_texting_phone':'Texting',    
                                             'theft':'Theft',
                                             'vehicle_drops_off_person':'Dropping off',
                                             'vehicle_picks_up_person':'Picking up',
                                             'vehicle_reversing':'Reversing', 
                                             'vehicle_starting':'Starting',
                                             'vehicle_stopping':'Stopping',
                                             'vehicle_turning_left':'Turning left',
                                             'vehicle_turning_right':'Turning right',
                                             'vehicle_u_turn':'Turning around'}
        
    def __repr__(self):
        return str('<vipy.dataset.meva: videos="%s", annotations="%s">' % (self.videodir, self.repodir))

    def activities(self):
        """Return a list of activities"""
        # This list in the repo is outdated
        # return sorted(list(readjson(os.path.join(self.repodir, 'annotation', 'DIVA-phase-2', 'activity-index.json')).keys()))        
        return sorted(list(self._d_category_to_shortlabel.keys()))

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
    
    def _parse_video(self, d_videoname_to_path, d_category_to_shortlabel, types_yamlfile, geom_yamlfile, act_yamlfile, stride=1, verbose=False):

        # Read YAML
        if verbose:
            print('[vipy.dataset.meva]: Parsing "%s"' % (act_yamlfile))
        geom_yaml = readyaml(geom_yamlfile)
        types_yaml = readyaml(types_yamlfile)
        act_yaml = readyaml(act_yamlfile)

        # Sanity check
        assert act_yamlfile.split('.')[:-2] == geom_yamlfile.split('.')[:-2], "Unmatched activity and geom yaml file"
        assert len(set([types_yaml[0]['meta'], geom_yaml[0]['meta'], act_yaml[0]['meta']]))==1, "Mismatched video name for '%s'" % act_yamlfile
        videoname = act_yaml[0]['meta']
        if videoname not in d_videoname_to_path:
            if verbose:
                warnings.warn('Invalid video "%s" in "%s" - Ignoring' % (videoname, filebase(act_yamlfile)))
            return None

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
            geom_yaml = sorted(geom_yaml, key=lambda x: int(x['ts0']))  # increasing
            for (k_geom, v) in enumerate(geom_yaml):
                if stride > 1 and k_geom > 0 and (k_geom < (len(geom_yaml)-stride)) and (k_geom % stride != 0):
                    continue  # Use vipy track interpolation to speed up parsing
                keyframe = int(v['ts0'])
                bb = [int(x) for x in v['g0'].split(' ')]
                bbox = BoundingBox(xmin=bb[0], ymin=bb[1], xmax=bb[2], ymax=bb[3])
                if not bbox.isvalid():
                    if verbose:
                        warnings.warn('Invalid bounding box: id1=%s, bbox="%s", file="%s" - Ignoring' % (str(v['id1']), str(bbox), filetail(geom_yamlfile)))
                elif v['id1'] not in d_id1_to_track:
                    d_id1_to_track[v['id1']] = Track(category=d_id1_to_category[v['id1']], framerate=framerate, keyframes=[keyframe], boxes=[bbox], boundary='strict')
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
                tracks = {d_id1_to_track[aid].id():d_id1_to_track[aid] for aid in actorid}
                vid.add(Activity(category=category, shortlabel=d_category_to_shortlabel[category],
                                 startframe=startframe, endframe=endframe, tracks=tracks, framerate=framerate, attributes={'src_status':v['act']['src_status']}))
            
        return vid
        
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


    def MEVA(self, stride=1, n_processes=1, verbose=True, n_videos=None, d_category_to_shortlabel=None):
        """Parse MEVA annotations from 'meva-data-repo/annotation/DIVA-phase-2/MEVA/meva-annotations/' into vipy.video.Scene()
       
        Kwiver packet format: https://gitlab.kitware.com/meva/meva-data-repo/blob/master/documents/KPF-specification-v4.pdf
          * Stride is an optional temporal step to skip bounding boxes annotated on every frame and use vipy.object.Track() linear interpolation between keyframes
          * d_category_to_shortlabel is a dictionary mapping category names to a short displayed label on the video.  The standard for visualization is that 
            tracked objects are displayed with their category label (e.g. 'Person', 'Vehicle'), and activities are labeled according to the set of objects that
            performing the activity.  When an activity occurs, the set of objects are labeled with the same color as 'Noun Verbing' (e.g. 'Person Entering', 
            'Person Reading', 'Vehicle Starting') where 'Verbing' is provided by the shortlabel.  

        """
        if verbose:
            num_yamlfiles = len(self._get_activities_yaml())
            print('[vipy.dataset.meva]: Loading %d YAML files (this takes a while)' % num_yamlfiles)

        d_category_to_shortlabel = d_category_to_shortlabel if d_category_to_shortlabel is not None else self._d_category_to_shortlabel
        d_videoname_to_path = {filebase(f):f for f in self._get_videos()}
        yamlfiles = zip(self._get_types_yaml(), self._get_geom_yaml(), self._get_activities_yaml())
        yamlfiles = list(yamlfiles)[0:n_videos] if n_videos is not None else list(yamlfiles)
        if n_processes > 1:
            from vipy.batch import Batch
            with Batch(list(yamlfiles), n_processes=n_processes) as batch:
                videolist = batch.map(lambda tga: self._parse_video(d_videoname_to_path, d_category_to_shortlabel, tga[0], tga[1], tga[2], stride=stride, verbose=verbose))
            return videolist
        else:
            return [self._parse_video(d_videoname_to_path, d_category_to_shortlabel, t, g, a, stride=stride, verbose=verbose) for (t,g,a) in yamlfiles]

        
    def analysis(self, meva=None, outdir=None):
        """Analyze the MEVA dataset to return helpful statistics and plots"""
        import matplotlib.pyplot as plt        
        
        videos = self.MEVA(stride=20) if meva is None else meva
        scenes = flatlist([m.activityclip() for m in meva if m is not None])
        activities = flatlist([s.activities().values() for s in scenes])
        tracks = flatlist([s.tracks().values() for s in scenes])
        outdir = tempdir() if outdir is None else outdir
        
        # Category distributions
        d = {}
        d['activity_categories'] = set([a.category() for a in activities])
        d['object_categories'] = set([t.category() for t in tracks])
        d['videos'] = set([v.filename() for v in videos if v is not None])
        d['num_activities'] = sorted([(k,len(v)) for (k,v) in groupbyasdict(activities, lambda a: a.category()).items()], key=lambda x: x[1])
        d['video_density'] = sorted([(v.filename(),len(v.activities())) for v in videos if v is not None], key=lambda x: x[1])
        
        # Histogram of instances
        (categories, freq) = zip(*reversed(d['num_activities']))
        barcolors = ['blue' if not 'vehicle' in c else 'green' for c in categories]
        d['num_activities_histogram'] = vipy.metrics.histogram(freq, categories, barcolors=barcolors, outfile=os.path.join(outdir, 'num_activities_histogram.pdf'), ylabel='Instances')

        # Scatterplot of box sizes
        (x, y, category) = zip(*[(max([t.meanshape()[1] for t in a.tracks().values()]), max([t.meanshape()[0] for t in a.tracks().values()]), a.category()) for a in activities])
        colors = colorlist()
        d_category_to_color = {c:colors[k % len(colors)] for (k,c) in enumerate(category)}
        plt.clf()
        plt.figure()
        plt.subplot(111)
        plt.grid(True)
        d_category_to_boxsizes = groupbyasdict(zip(x,y,category), lambda xyc: xyc[2])
        for c in categories:
            (xc,yc) = zip(*[(xx,yy) for (xx,yy,cc) in d_category_to_boxsizes[c]])
            plt.scatter(xc, yc, c=d_category_to_color[c], label=c)
        plt.xlabel('Bounding box (width)')
        plt.ylabel('Bounding box (height)')
        plt.gca().set_axisbelow(True)                
        plt.axis([0, max(max(x),max(y)), 0, max(max(x),max(y))])
        lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        d['bounding_box_scatterplot'] = os.path.join(outdir, 'bounding_box_scatterplot.pdf')
        plt.savefig(d['bounding_box_scatterplot'], bbox_extra_artists=(lgd,), bbox_inches='tight')

        # Separate scatterplots of box sizes
        (x, y, category) = zip(*[(max([t.meanshape()[1] for t in a.tracks().values()]), max([t.meanshape()[0] for t in a.tracks().values()]), a.category()) for a in activities])
        colors = colorlist()
        d_category_to_color = {c:colors[k % len(colors)] for (k,c) in enumerate(category)}
        d_category_to_boxsizes = groupbyasdict(zip(x,y,category), lambda xyc: xyc[2])
        d['category_bounding_box_scatterplot'] = []
        for c in categories:
            plt.clf()
            plt.figure()
            plt.subplot(111)
            plt.grid(True)            
            (xc,yc) = zip(*[(xx,yy) for (xx,yy,cc) in d_category_to_boxsizes[c]])
            plt.scatter(xc, yc, c=d_category_to_color[c], label=c)
            plt.xlabel('Bounding box (width)')
            plt.ylabel('Bounding box (height)')
            plt.legend()            
            plt.gca().set_axisbelow(True)                
            plt.axis([0, max(max(x),max(y)), 0, max(max(x),max(y))])
            filename = os.path.join(outdir, 'category_bounding_box_scatterplot_%s.pdf' % c)
            d['category_bounding_box_scatterplot'].append(filename)
            plt.savefig(filename)
        
        # 2D histogram of box sixes
        plt.clf()
        plt.figure()
        plt.hist2d(x, y, bins=10)
        plt.xlabel('Bounding box (width)')
        plt.ylabel('Bounding box (height)')        
        d['2D_bounding_box_histogram'] = os.path.join(outdir, '2D_bounding_box_histogram.pdf')
        plt.savefig(d['2D_bounding_box_histogram'])
        
        # Scatterplot of people and vehicles box sizes
        plt.clf()
        plt.figure()
        plt.grid(True)
        d_category_to_color = {'person':'blue', 'vehicle':'green'}
        for c in ['person', 'vehicle']:
            (xc, yc) = zip(*[(t.meanshape()[1], t.meanshape()[0]) for t in tracks if t.category() == c])
            plt.scatter(xc, yc, c=d_category_to_color[c], label=c)
        plt.xlabel('bounding box (width)')
        plt.ylabel('bounding box (height)')
        plt.axis([0, max(max(x),max(y)), 0, max(max(x),max(y))])                
        plt.legend()
        plt.gca().set_axisbelow(True)        
        d['object_bounding_box_scatterplot'] = os.path.join(outdir, 'object_bounding_box_scatterplot.pdf')
        plt.savefig(d['object_bounding_box_scatterplot'])
        
        # 2D histogram of people and vehicles box sizes
        for c in ['person', 'vehicle']:
            (x, y) = zip(*[(t.meanshape()[1], t.meanshape()[0]) for t in tracks if t.category() == c])
            plt.clf()
            plt.figure()
            plt.hist2d(x, y, bins=10)
            plt.xlabel('Bounding box (width)')
            plt.ylabel('Bounding box (height)')
            
            d['2D_%s_bounding_box_histogram' % c] = os.path.join(outdir, '2D_%s_bounding_box_histogram.pdf' % c)
            plt.savefig(d['2D_%s_bounding_box_histogram' % c])

        return d
