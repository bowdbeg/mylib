from collections import defaultdict
import xml.etree.ElementTree as ET
from relation_data import RelationDatum
import re


class TimebankDatum(RelationDatum):
    def __init__(self, file) -> None:
        self.data = self.parse(file)

    def parse(self, file):
        tree = ET.parse(file)
        root = tree.getroot()

        txtnode = tree.find("TEXT")
        events = defaultdict(dict)
        timexs = defaultdict(dict)
        text = txtnode.text.replace("\n", "")
        for n in txtnode:
            if n.tag == "EVENT":
                idx = n.attrib["eid"]
                events[idx]["class"] = n.attrib["class"]
                events[idx]["text"] = n.text
                events["start"] = len(text)
                text += n.text
                events["end"] = len(text)
                text += n.tail

            elif "TIMEX" in n.tag:
                timetag = n.tag
                idx = n.attrib["tid"]
                timexs[idx]["type"] = n.attrib["type"]
                timexs[idx]["value"] = n.attrib["value"]
                timexs[idx]["text"] = n.text
                timexs["start"] = len(text)
                text += n.text
                timexs["end"] = len(text)
                text += n.tail
            else:
                raise NotImplementedError

        # entities outside of text
        for n in tree.findall("//{}".format(timetag)):
            if not n.attrib["tid"] in timexs:
                idx = n.attrib["tid"]
                timexs[idx]["type"] = n.attrib["type"]
                timexs[idx]["value"] = n.attrib["value"]
                timexs[idx]["text"] = n.text
                timexs[idx]["start"] = None
                timexs[idx]["end"] = None
        for n in tree.findall("//{}".format("EVENT")):
            if not n.attrib["eid"] in events:
                idx = n.attrib["eid"]
                events[idx]["class"] = n.attrib["class"]
                events[idx]["text"] = n.text
                events[idx]["start"] = None
                events[idx]["end"] = None

        # instances
        instances = defaultdict(dict)
        instances.update(timexs)
        instances.update(events)
        for n in tree.findall('//{}'.format('MAKEINSTANCE')):
            idx = n.attrib['eiid']
            instances[idx]['eid'] = n.attrib['eventID']
            instances[idx]['aspect'] = n.attrib['aspect']
            instances[idx]['polarity'] = n.attrib['polarity']
            instances[idx]['pos'] = n.attrib['pos']
            if 'modality' in n.attrib:
                instances[idx]['modality'] = n.attrib['modality']
            instances[n.attrib['eventID']]['eiid'] = idx
        
        tlinks = defaultdict(dict)
        for n in tree.findall('//{}'.format('TLINK')):
            idx = n.attrib['lid']
            instances[idx]['type'] = n.attrib['relType']
            instances[idx]['from'] = n.attrib['timeID'] if 'timeID' in n.attrib else n.attrib['eventInstanceID']
            instances[idx]['to'] = n.attrib['timeID'] if 'timeID' in n.attrib else n.attrib['eventInstanceID']
        
        return {'event': events, 'text':text, 'timex':timexs, 'tlink':tlinks, 'instance':instances}

