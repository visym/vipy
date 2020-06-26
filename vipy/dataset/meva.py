import os
from vipy.util import remkdir, readjson, readyaml, findyaml, findvideo, filetail, findjson, filebase, readlist, groupbyasdict, save, flatlist, isstring, tempdir, readcsv, delpath, temphtml, tolist
from vipy.video import VideoCategory, Scene
from vipy.object import Track
from vipy.activity import Activity
from vipy.geometry import BoundingBox
from vipy.batch import Batch
from vipy.show import colorlist
import numpy as np
import warnings
import shutil
import vipy.globals


class KF1(object):
    def __init__(self, videodir, repodir, contrib=False, stride=1, verbose=True, n_videos=None, withprefix=None, d_category_to_shortlabel=None, merge=False, actor=False, disjoint=False, unpad=False):
        """Parse MEVA annotations (http://mevadata.org) for KNown Facility 1 dataset into vipy.video.Scene() objects
       
        Kwiver packet format: https://gitlab.kitware.com/meva/meva-data-repo/blob/master/documents/KPF-specification-v4.pdf
        Inputs:
          -videodir=str:  path to Directory containing 'drop-01' 
          -repodir=str:  path to directory containing clone of https://gitlab.kitware.com/meva/meva-data-repo
          -stride=int: the temporal stride in frames for importing bounding boxes, vipy will do linear interpoluation and boundary handling
          -n_videos=int:  only return an integer number of videos, useful for debugging or for previewing dataset
          -withprefix=list:  only return videos with the filename containing one of the strings in withprefix list, useful for debugging
          -contrib=bool:  include the noisy contrib anntations from DIVA performers
          -d_category_to_shortlabel is a dictionary mapping category names to a short displayed label on the video.  The standard for visualization is that 
            tracked objects are displayed with their category label (e.g. 'Person', 'Vehicle'), and activities are labeled according to the set of objects that
            performing the activity.  When an activity occurs, the set of objects are labeled with the same color as 'Noun Verbing' (e.g. 'Person Entering', 
            'Person Reading', 'Vehicle Starting') where 'Verbing' is provided by the shortlabel.   This is optional, and will use the default mapping if None
          -verbose=bool:  Parsing verbosity
          -merge:  deduplicate annotations for each video across YAML files by merging them by mean spatial IoU per track (>0.5) and temporal IoU (>0)
          -actor [bool]:  Include only those activities that include an associated track for the primary actor: "Person" for "person_*" and "hand_*", else "Vehicle"
          -disjoint [bool]:  Enforce that overlapping causal activities (open/close, enter/exit, ...) are disjoint for a track
          -unpad [bool]:  remove the arbitrary padding assigned during dataset creation

        """
        
        self.videodir = videodir
        self.repodir = repodir

        assert os.path.exists(os.path.join(self.videodir, 'drop-01')), "Invalid input - videodir '%s' must contain the drop-01, drop-02 and drop-03 subdirectories.  See http://mevadata.org/#getting-data" % videodir
        assert os.path.exists(os.path.join(self.repodir, 'annotation')), "Invalid input - repodir '%s' must contain the clone of https://gitlab.kitware.com/meva/meva-data-repo" % repodir

        # Shortlabels are optional and used for showing labels on videos only
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
        self._d_oldcategory_to_newcategory = {k:v for (k,v) in readcsv(os.path.join(self.repodir, 'documents', 'activity-name-mapping.csv'))[1:]}

        d_category_to_shortlabel = d_category_to_shortlabel if d_category_to_shortlabel is not None else self._d_category_to_shortlabel
        d_videoname_to_path = {filebase(f):f for f in self._get_videos()}
        yamlfiles = zip(self._get_types_yaml(), self._get_geom_yaml(), self._get_activities_yaml())
        yamlfiles = [y for y in yamlfiles if contrib is True or 'contrib' not in y[0]]
        yamlfiles = list(yamlfiles)[0:n_videos] if n_videos is not None else list(yamlfiles)
        if withprefix is not None:            
            yamlfiles = [y for y in yamlfiles if any([(p in y[0]) for p in tolist(withprefix)])]  

        if verbose:
            print('[vipy.dataset.meva.KF1]: Loading %d YAML files' % len(yamlfiles))
            if len(yamlfiles) > 100 and vipy.globals.max_workers() == 1: 
                print('[vipy.dataset.meva.KF1]: This takes a while since parsing YAML files in python is painfully slow, consider calling "vipy.globals.max_workers(n)" for n>1 before loading the dataset for parallel parsing')

        # Parallel video annotation
        if vipy.globals.max_workers() > 1:
            from vipy.batch import Batch
            self._vidlist = Batch(list(yamlfiles)).map(lambda tga: self._parse_video(d_videoname_to_path, d_category_to_shortlabel, tga[0], tga[1], tga[2], stride=stride, verbose=verbose, actor=actor))
        else:
            self._vidlist = [self._parse_video(d_videoname_to_path, d_category_to_shortlabel, t, g, a, stride=stride, verbose=verbose, actor=actor) for (t,g,a) in yamlfiles]
        self._vidlist = [v for v in self._vidlist if v is not None]

        # Merge and dedupe activities and tracks across YAML files for same video, using temporal and spatial IoU association.
        #   The MEVA dataset is "activity-centric" so that each activity is labeled independently.  There may be tracks in the dataset
        #   that are the same instance in the video, but are different track IDs in the dataset.  The result is disjoint activity labels 
        #   in a non-disjoint activity in a video.  Yuck..  Try to merge them.  This is experimental, since it tries to use IoU for merging, 
        #   which does not work in general.  This requires global track correspondence.  
        if merge:
            print('[vipy.dataset.meva.KF1]: merging videos ...')
            self._vidlist = [v[0].clone().union(v[1:]) for (f, v) in groupbyasdict([a for vid in self._vidlist for a in vid.activitysplit()], lambda s: s.filename()).items()]
        
        # Enforce disjoint causal activities
        #   Due to the arbitrary temporal padding in the annotation definitions, merged causal activiites can overlap
        #   Enforce that causal activities (open/close, enter/exit, pickup/putdown, load/unload) for the same track are disjoint
        if disjoint:
            V = self._vidlist
            V = [v.activitymap(lambda a: a.disjoint([sa for sa in v.activitylist() if sa.category() == 'person_closes_vehicle_door']) if a.category() == 'person_opens_vehicle_door' else a) for v in V]  
            V = [v.activitymap(lambda a: a.disjoint([sa for sa in v.activitylist() if sa.category() == 'person_closes_vehicle_trunk']) if a.category() == 'person_opens_vehicle_trunk' else a) for v in V]  
            V = [v.activitymap(lambda a: a.disjoint([sa for sa in v.activitylist() if sa.category() == 'person_exits_vehicle']) if a.category() == 'person_enters_vehicle' else a) for v in V]  
            V = [v.activitymap(lambda a: a.disjoint([sa for sa in v.activitylist() if sa.category() == 'person_enters_scene_through_structure']) if a.category() == 'person_exits_scene_through_structure' else a) for v in V]  
            V = [v.activitymap(lambda a: a.disjoint([sa for sa in v.activitylist() if sa.category() == 'person_closes_facility_door']) if a.category() == 'person_opens_facility_door' else a) for v in V]  
            V = [v.activitymap(lambda a: a.disjoint([sa for sa in v.activitylist() if sa.category() == 'person_unloads_vehicle']) if a.category() == 'person_loads_vehicle' else a) for v in V]  
            V = [v.activitymap(lambda a: a.disjoint([sa for sa in v.activitylist() if sa.category() == 'person_puts_down_object']) if a.category() == 'person_picks_up_object' else a) for v in V]  
            V = [v.activityfilter(lambda a: len(a)>0) for v in V]  # some activities may be zero length after disjoint
            self._vidlist = V

        # Remove the arbitrary temporal padding applied during dataset creation
        if unpad:
            # MEVA annotations assumptions:  https://docs.google.com/spreadsheets/d/19I3C5Zb6RHS0QC30nFT_m0ymArzjvlPLfb5SSRQYLUQ/edit#gid=0
            # Pad one second before, zero seconds after
            before1after0 = set(['person_opens_facility_door', 'person_closes_facility_door', 'person_opens_vehicle_door', 'person_closes_vehicle_door', 
                                 'person_opens_trunk', 'person_closes_trunk', 'vehicle_stops', 'person_interacts_with_laptop'])        
            V = [v.activitymap(lambda a: a.temporalpad( (-v.framerate()*1.0, 0) ) if a.category() in before1after0 else a) for v in self._vidlist]

            # pad one second before, one second after, up to maximum of two seconds
            before1after1max2 = set(['person_enters_scene_through_structure'])
            V = [v.activitymap(lambda a: a.temporalpad(max(0, -v.framerate()*1.0)) if a.category() in before1after1max2 else a) for v in V]

            # person_exits_scene_through_structure:  Pad one second before person_opens_facility_door label (if door collection), and ends with enough padding to make this minimum two seconds
            V = [v.activitymap(lambda a: a.temporalpad( (-v.framerate()*1.0, 0) ) if a.category() == 'person_exits_scene_through_structure' else a) for v in V]        
        
            # person_enters_vehicle: Starts one second before person_opens_vehicle_door activity label and ends at the end of person_closes_vehicle_door activity
            V = [v.activitymap(lambda a: a.temporalpad( (-v.framerate()*1.0, 0) ) if a.category() == 'person_enters_vehicle' else a) for v in V]        

            # person_exits_vehicle:  Starts one second before person_opens_vehicle_door, and ends at person_exits_vehicle with enough padding to make this minimum two seconds
            V = [v.activitymap(lambda a: a.temporalpad( (-v.framerate()*1.0, 0) ) if a.category() == 'person_exits_vehicle' else a) for v in V]        

            # person_unloads_vehicle:  one second of padding before cargo starts to move
            V = [v.activitymap(lambda a: a.temporalpad( (-v.framerate()*1.0, 0) ) if a.category() == 'person_unloads_vehicle' else a) for v in V]        

            # person_talks_to_person:  Equal padding to minimum of five seconds            
            # person_texting_on_phone:  Equal padding to minimum of two seconds

            # Pad one second before, one second after
            before1after1 = set(['vehicle_turns_left', 'vehicle_turns_right', 'person_transfers_object',
                                 'person_sets_down_object', 'hand_interacts_with_person', 'person_embraces_person', 'person_purchases',
                                 'vehicle_picks_up_person', 'vehicle_drops_off_person'])
            V = [v.activitymap(lambda a: a.temporalpad(-v.framerate()*1.0) if a.category() in before1after1 else a) for v in V]

            # Pad zero second before, one second after
            before0after1 = set(['vehicle_makes_u_turn', 'person_picks_up_object'])
            V = [v.activitymap(lambda a: a.temporalpad( (0, -v.framerate()*1.0) ) if a.category() in before0after1 else a) for v in V]

            # person_abandons_package:  two seconds before, two seconds after
            V = [v.activitymap(lambda a: a.temporalpad(-v.framerate()*2.0) if a.category() == 'person_abandons_package' else a) for v in V]            
            self._vidlist = V

        # Remove empty tracks and activities
        self._vidlist = [v.trackfilter(lambda t: len(t) > 0) for v in self._vidlist]
        self._vidlist = [v.activityfilter(lambda a: len(a) > 0) for v in self._vidlist]

    def __getitem__(self, k):
        return self._vidlist[k]

    def __iter__(self):
        for v in self._vidlist:
            yield v

    def __len__(self):
        return len(self._vidlist)
                
    def __repr__(self):
        return str('<vipy.dataset.meva.KF1: videos=%d, videodir="%s", annotationdir="%s">' % (len(self), self.videodir, self.repodir))

    def _activities_to_required_objects(self):
        """Return a dictionary of activity keys to set of required objects.  This is currently wrong."""
        raise ValueError('This mapping is currently wrong in the Kitware repository')
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
    
    def _parse_video(self, d_videoname_to_path, d_category_to_shortlabel, types_yamlfile, geom_yamlfile, act_yamlfile, stride=1, verbose=False, actor=False):
        """Reference: https://gitlab.kitware.com/meva/meva-data-repo/-/blob/master/documents/KPF-specification-v4.pdf"""
        
        # Read YAML
        if verbose:
            print('[vipy.dataset.meva.KF1]: Parsing "%s"' % (act_yamlfile))
        geom_yaml = readyaml(geom_yamlfile)
        types_yaml = readyaml(types_yamlfile)
        act_yaml = readyaml(act_yamlfile)

        # Sanity check
        assert act_yamlfile.split('.')[:-2] == geom_yamlfile.split('.')[:-2], "Unmatched activity and geom yaml file"
        assert len(set([types_yaml[0]['meta'], geom_yaml[0]['meta'], act_yaml[0]['meta']]))==1, "Mismatched video name for '%s'" % act_yamlfile
        videoname = act_yaml[0]['meta'] if act_yaml[0]['meta'][-4:] != '.avi' else act_yaml[0]['meta'][0:-4]  # strip .avi 
        if videoname not in d_videoname_to_path:
            if verbose:
                print('[vipy.dataset.meva.KF1]: Invalid MEVA video "%s" in "%s" - Ignoring' % (videoname, filebase(act_yamlfile)))
            return None

        # Parse video
        framerate = 30.0  # All videos are universally 30Hz (from Roddy)
        vid = Scene(filename=d_videoname_to_path[videoname], framerate=framerate)


        # Parse tracks        
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
                        print('[vipy.dataset.meva.KF1]: Invalid bounding box: id1=%s, bbox="%s", file="%s" - Ignoring' % (str(v['id1']), str(bbox), delpath(self.repodir, geom_yamlfile)))
                elif v['id1'] not in d_id1_to_track:
                    d_id1_to_track[v['id1']] = Track(category=d_id1_to_category[v['id1']], framerate=framerate, keyframes=[keyframe], boxes=[bbox], boundary='strict')
                else:
                    d_id1_to_track[v['id1']].add(keyframe=keyframe, box=bbox)
                
        # Add tracks to scene
        for (k,v) in d_id1_to_track.items():
            try:
                vid.add(v, rangecheck=True)  # throw exception if all tracks are outside the image rectangle
            except Exception as e:
                print('[vipy.dataset.meva.KF1]: track import error "%s" for trackid=%s, track=%s - SKIPPING' % (str(e), k, str(v)))

        # Category to actor:  This defines the primary role for the activity (for tube based representations)
        f_activity_to_actor = lambda c: 'Person' if (c.split('_')[0] == 'person' or 'hand' in c) else 'Vehicle'
        
        # Parse activities
        for v in act_yaml:
            if 'act' in v:
                if 'act2' in v['act']:
                    act2 = v['act']['act2']
                    if isinstance(act2, set):
                        category = list(act2)[0]
                    elif isinstance(act2, dict):
                        category = list(act2.keys())[0]
                    else:
                        raise ValueError('YAML parsing error for "%s"' % str(act2))
                elif 'act3' in v['act']:
                    act3 = v['act']['act3']
                    if isinstance(act3, set):
                        category = list(act3)[0]
                    elif isinstance(act3, dict):
                        category = list(act3.keys())[0]
                    else:
                        raise ValueError('YAML parsing error for "%s"' % str(act3))
                else:
                    raise ValueError('Invalid activity YAML - act2 or act3 must be specified')
                assert len(v['act']['timespan']) == 1, "Multi-span activities not parsed"

                if category not in self.categories():
                    if category in self._d_oldcategory_to_newcategory:
                        category = self._d_oldcategory_to_newcategory[category]  # rationalize
                    else:
                        raise ValueError('undefined category "%s"' % category)
                
                startframe = int(v['act']['timespan'][0]['tsr0'][0])
                endframe = int(v['act']['timespan'][0]['tsr0'][1])                
                actorid = [x['id1'] for x in v['act']['actors']]   
                if actor:                    
                    nounid = [d_id1_to_track[a].id() for a in actorid if f_activity_to_actor(category).lower() == d_id1_to_track[a].category().lower()]
                    if len(nounid) == 0:
                        print('[vipy.dataset.meva.KF1]: activity "%s" without a required primary actor "%s" - SKIPPING' % (category, f_activity_to_actor(category)))
                        continue
                    nounid = nounid[0] if len(nounid) > 0 else None   # first track in activity of required object class for this category is assumed to be the performer/actor/noun

                for aid in actorid:
                    if not aid in d_id1_to_track:
                        print('[vipy.dataset.meva.KF1]: ActorID %d referenced in activity yaml "%s" not found in geom yaml "%s" - Skipping' % (aid, delpath(self.repodir, act_yamlfile), delpath(self.repodir, geom_yamlfile)))
                
                # Add activity to scene:  include YAML file details in activity attributes for provenance if there are labeling bugs
                tracks = {d_id1_to_track[aid].id():d_id1_to_track[aid] for aid in actorid if aid in d_id1_to_track}  # order preserving (python 3.6)
                if len(tracks) > 0:
                    try:
                        vid.add(Activity(category=category, shortlabel=d_category_to_shortlabel[category], actorid=nounid if actor else None,
                                         startframe=startframe, endframe=endframe, tracks=tracks, framerate=framerate, 
                                         attributes={'act':v['act'], 'act_yaml':act_yamlfile, 'geom_yaml':geom_yamlfile}), rangecheck=True)
                    except Exception as e:
                        print('[vipy.dataset.meva.KF1]: activity import error "%s" for activity="%s" - SKIPPING' % (str(e), str(v)))
            
        return vid


    def videos(self):
        """Return list of activity videos"""
        return [v for v in self._vidlist if v is not None]

    def tolist(self):
        return self.videos()

    def instances(self, padframes=0):
        """Return list of activity instances"""
        if vipy.globals.max_workers() > 1:
            return [a for A in Batch(self.videos()).activityclip(padframes=padframes) for a in A]
        else:
            warnings.warn('Consider setting vipy.globals.max_workers(n) for n>1 to speed this up')
            return [a for v in self.videos() for a in v.activityclip(padframes=padframes)]
        

    def categories(self):
        """Return a list of activity categories"""
        return sorted(list(self._d_category_to_shortlabel.keys()))
        
    def analysis(self, outdir=None):
        """Analyze the MEVA dataset to return helpful statistics and plots"""
        import matplotlib.pyplot as plt        
        import vipy.metrics
        
        videos = self._vidlist
        scenes = flatlist([m.activityclip() for m in videos if m is not None])
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
        colors = colorlist()

        # Scatterplot of people and vehicles box sizes
        (x, y) = zip(*[(t.meanshape()[1], t.meanshape()[0]) for t in tracks])
        plt.clf()
        plt.figure()
        plt.grid(True)
        d_category_to_color = {'person':'blue', 'vehicle':'green'}
        for c in ['person', 'vehicle']:
            (xc, yc) = zip(*[(t.meanshape()[1], t.meanshape()[0]) for t in tracks if t.category() == c])
            plt.scatter(xc, yc, c=d_category_to_color[c], label=c)
        plt.xlabel('bounding box (width)')
        plt.ylabel('bounding box (height)')
        plt.axis([0, 1000, 0, 1000])                
        plt.legend()
        plt.gca().set_axisbelow(True)        
        d['object_bounding_box_scatterplot'] = os.path.join(outdir, 'object_bounding_box_scatterplot.pdf')
        plt.savefig(d['object_bounding_box_scatterplot'])
        
        # 2D histogram of people and vehicles box sizes
        for c in ['person', 'vehicle']:
            (xc, yc) = zip(*[(t.meanshape()[1], t.meanshape()[0]) for t in tracks if t.category() == c])
            plt.clf()
            plt.figure()
            plt.hist2d(xc, yc, bins=10)
            plt.xlabel('Bounding box (width)')
            plt.ylabel('Bounding box (height)')
            
            d['2D_%s_bounding_box_histogram' % c] = os.path.join(outdir, '2D_%s_bounding_box_histogram.pdf' % c)
            plt.savefig(d['2D_%s_bounding_box_histogram' % c])

        # Mean track size per activity category
        d_category_to_xy = {k:np.mean([t.meanshape() for v in vlist for t in v.tracklist()], axis=0) for (k,vlist) in groupbyasdict(scenes, lambda v: v.category()).items()}        
        plt.clf()
        plt.figure()
        plt.grid(True)
        d_category_to_color = {c:colors[k % len(colors)] for (k,c) in enumerate(d_category_to_xy.keys())}
        for c in d_category_to_xy.keys():
            (xc, yc) = d_category_to_xy[c]
            plt.scatter(xc, yc, c=d_category_to_color[c], label=c)
        plt.xlabel('bounding box (width)')
        plt.ylabel('bounding box (height)')
        plt.axis([0, 600, 0, 600])                
        plt.gca().set_axisbelow(True)        
        lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        d['activity_bounding_box_scatterplot'] = os.path.join(outdir, 'activity_bounding_box_scatterplot.pdf')
        plt.savefig(d['activity_bounding_box_scatterplot'], bbox_extra_artists=(lgd,), bbox_inches='tight')
    
        return d

    def review(self, outfile=None, mindim=512):        
        """Generate a standalone HTML file containing quicklooks for each annotated activity in dataset, along with some helpful provenance information for where the annotation came from"""
        if vipy.globals.max_workers() == 1:
            warnings.warn("Generating review HTML is very time consuming, consider setting vipy.global.max_workers(n) for n > 1 for parallel video processing")
        quicklist = Batch(self._vidlist).map(lambda v: [(c.load().quicklook(context=True), c.flush()) for c in v.mindim(512).activityclip()])
        quicklooks = [imq for q in quicklist for (imq, c) in q]  # for HTML display purposes
        provenance = [{'clip':str(c), 'activities':str(';'.join([str(a) for a in c.activitylist()])), 'category':c.category(), 'yamlfile':c.activitylist()[0].attributes['act_yaml']} for q in quicklist for (imq, c) in q]
        (quicklooks, provenance) = zip(*sorted([(q,p) for (q,p) in zip(quicklooks, provenance)], key=lambda x: x[1]['category']))  # sorted in category order
        return vipy.visualize.tohtml(quicklooks, provenance, title='MEVA-KF1 annotation quicklooks', outfile=outfile, mindim=mindim)
