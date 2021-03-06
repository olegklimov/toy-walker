import csv, sys, os
import numpy as np

html1 = """
<html>
<script type="text/javascript" src="https://www.gstatic.com/charts/loader.js"></script>
<body>
<div id="chart_div" style="width: 100%; height: 100%;"></div>
<script>
google.charts.load('current', {packages: ['corechart', 'line']});
google.charts.setOnLoadCallback(drawBasic);

function drawBasic() {

      var data = new google.visualization.DataTable();
      data.addColumn('number', 'TimestepsSoFar');
      //data.addColumn('number', 'loss_pol_surr');
      data.addColumn('number', 'loss_vf_loss');
      data.addColumn('number', 'loss_kl');
      data.addColumn('number', 'ev_tdlam_before')
      data.addColumn('number', 'EpRewMean');
      data.addColumn('number', 'EpLenMean');
      data.addColumn('number', 'TimeElapsed');
      data.addRows([
"""

html2 = """
] );
      var options = {
        series: {
          0: {targetAxisIndex: 0},
          1: {targetAxisIndex: 0},
          2: {targetAxisIndex: 0},
          3: {targetAxisIndex: 1},
          4: {targetAxisIndex: 1},
          5: {targetAxisIndex: 2},
        },
        hAxis: {
            viewWindow: {
                min: 0,
                max: 30000000
            }
        },
        vAxes: {
            0: {
                minValue: 0,
                maxValue: 1
            },
            1: {
                minValue: -1000,
                maxValue:  1000
            },
            2: {
                gridlines: { count: 0 },
                minValue: 0,
                maxValue: 3*3600
            }
        }
      };
      var chart = new google.visualization.LineChart(document.getElementById('chart_div'));
      chart.draw(data, options);
    }
</script>
</body>
</html>
"""


need_files = []
for root, dirs, files in os.walk("."):
    for fn in files:
        base,ext = os.path.splitext(fn)
        if ext=='.csv':
            fn = root + "/" + base
            need_files.append(fn)

for fn in need_files:
    fn_html = "%s.html" % fn
    fn_csv  = "%s.csv" % fn
    print("file://%s/%s.html" % (os.getcwd(), fn))
    try:
        if os.path.getmtime(fn_html) >= os.path.getmtime(fn_csv):
            continue
    except: pass
    print("CONVERT")
    with open(fn_html, "w") as w:
        w.write(html1)
        with open(fn_csv, newline='') as f:
            #reader = csv.reader(f)
            reader = csv.DictReader(f)
            for row in reader:
                w.write("[%s],\n" %
                    (",".join([
                    row["TimestepsSoFar"],
                    #row["loss_pol_surr"],
                    str(np.clip(float(row["loss_vf_loss"]), 0,1)),
                    str(np.clip(float(row["loss_kl"]), 0,1)),
                    str(np.clip(float(row["ev_tdlam_before"]), 0,1)),
                    row["EpRewMean"],
                    row["EpLenMean"],
                    row["TimeElapsed"],
                    ])
                    ))
        w.write(html2)
