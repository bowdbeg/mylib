from collections import defaultdict
import xml.etree.ElementTree as ET
import re
from pathlib import Path


class TimebankDatum:
    def __init__(self, file) -> None:
        self.file = Path(file)
        self.raw = self.file.read_text()
        self.tree = ET.fromstring(self.raw)

        self.__dict__.update(self.parse(self.tree))

    def parse(self, tree):
        txtnode = tree.find("TEXT")
        events = defaultdict(dict)
        timexs = defaultdict(dict)
        text = txtnode.text.replace("\n", "")
        for n in txtnode:
            if n.tag == "EVENT":
                idx = n.attrib["eid"]
                events[idx]["class"] = n.attrib["class"]
                events[idx]["text"] = n.text
                events[idx]["start"] = len(text)
                text += n.text
                events[idx]["end"] = len(text)
                text += n.tail

            elif "TIMEX" in n.tag:
                timetag = n.tag
                idx = n.attrib["tid"]
                timexs[idx]["class"] = n.attrib["type"]
                timexs[idx]["value"] = n.attrib["value"]
                timexs[idx]["text"] = n.text
                timexs[idx]["start"] = len(text)
                text += n.text
                timexs[idx]["end"] = len(text)
                text += n.tail
            else:
                raise NotImplementedError

        # entities outside of text
        for n in tree.findall(".//DCT/{}".format(timetag)):
            if not n.attrib["tid"] in timexs:
                idx = n.attrib["tid"]
                timexs[idx]["class"] = n.attrib["type"]
                timexs[idx]["value"] = n.attrib["value"]
                timexs[idx]["text"] = n.text
                timexs[idx]["start"] = None
                timexs[idx]["end"] = None
        # for n in tree.findall(".//{}".format("EVENT")):
        #     if not n.attrib["eid"] in events:
        #         idx = n.attrib["eid"]
        #         events[idx]["class"] = n.attrib["class"]
        #         events[idx]["text"] = n.text
        #         events[idx]["start"] = None
        #         events[idx]["end"] = None

        # instances
        eid2eiid = dict()
        eiid2eid = dict()
        for n in tree.findall(".//{}".format("MAKEINSTANCE")):
            eid = n.attrib["eventID"]
            eiid = n.attrib["eiid"]
            eid2eiid[eid] = eiid
            eiid2eid[eiid] = eid

            events[eid]["eiid"] = eiid
            events[eid]["aspect"] = n.attrib["aspect"]
            if "polarity" in n.attrib:
                events[eid]["polarity"] = n.attrib["polarity"]
            if 'pos' in n.attrib:
                events[eid]["pos"] = n.attrib["pos"]
            if "modality" in n.attrib:
                events[eid]["modality"] = n.attrib["modality"]

        entities = dict()
        entities.update(timexs)
        entities.update(events)

        links = defaultdict(dict)
        for n in tree.findall(".//{}".format("TLINK")):
            idx = n.attrib["lid"]
            links[idx]["class"] = n.attrib["relType"]
            links[idx]["type"] = "TLINK"
            links[idx]["head"] = n.attrib["timeID"] if "timeID" in n.attrib else n.attrib["eventInstanceID"]
            links[idx]["tail"] = n.attrib["relatedToTime"] if "relatedToTime" in n.attrib else n.attrib["relatedToEventInstance"]
        for n in tree.findall(".//{}".format("SLINK")):
            idx = n.attrib["lid"]
            links[idx]["class"] = n.attrib["relType"]
            links[idx]["type"] = "SLINK"
            links[idx]["head"] = n.attrib["timeID"] if "timeID" in n.attrib else n.attrib["eventInstanceID"]
            if 'relatedToTime' in n.attrib:
                links[idx]["tail"] = n.attrib["relatedToTime"]
            elif 'relatedToEventInstance' in n.attrib:
                links[idx]['tail']=n.attrib["relatedToEventInstance"]
            elif 'subordinatedEventInstance' in n.attrib:
                links[idx]['tail']=n.attrib['subordinatedEventInstance']
            else:
                raise NotImplementedError

        return {"entity": entities, "relation": links, "eid2eiid": eid2eiid, "eiid2eid": eiid2eid, "text": text}

    def export(self, ofile=None):
        txt = self.raw
        txt = re.sub("<.?LINK.*/>", "", txt).replace("</TimeML>", "").strip()
        lines = []
        for lid, v in self.relation.items():
            hid = v["head"]
            tid = v["tail"]
            hnm = "eventInstanceID" if hid[0] == "e" else "timeID"
            tnm = "relatedToEventInstance" if tid[0] == "e" else "relatedToTime"
            l = '<TLINK lid="{}" relType="{}" {}="{}" {}="{}" />'.format(lid, v["class"], hnm, hid, tnm, tid)
            lines.append(l)
        lines.append("</TimeML>")
        txt += "\n".join(lines)

        if ofile:
            ofile.write_text(txt)
        return txt

    def reset_relation(self):
        self.relation = defaultdict(dict)

    def add_relation(self, head, tail, label):
        hid = head if head[:2] == "ei" or head[0] == "t" else self.eid2eiid[head]
        tid = tail if tail[:2] == "ei" or tail[0] == "t" else self.eid2eiid[tail]

        key = "l{}".format(len(self.relation))
        self.relation[key]["class"] = label
        self.relation[key]["head"] = hid
        self.relation[key]["tail"] = tid
        self.relation[key]["type"] = "TLINK"
