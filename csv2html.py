import csv, sys, os

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
      data.addColumn('number', 'EpRewMean');
      data.addColumn('number', 'EpLenMean');
      data.addRows([
"""

html2 = """
] );
      var options = {
        series: {
          0: {targetAxisIndex: 0},
          1: {targetAxisIndex: 0},
          2: {targetAxisIndex: 1},
          3: {targetAxisIndex: 1},
        },
        hAxis: {
            viewWindow: {
                min: 0,
                max: 8000000
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
for root, dirs, files in os.walk("progress"):
    for fn in files:
        base,ext = os.path.splitext(fn)
        if ext=='.csv':
            fn = root + "/" + base
            need_files.append(fn)

for fn in need_files:
    with open("%s.html" % fn, "w") as w:
        w.write(html1)
        with open('%s.csv' % fn, newline='') as f:
            #reader = csv.reader(f)
            reader = csv.DictReader(f)
            for row in reader:
                w.write("[%s],\n" %
                    (",".join([
                    row["TimestepsSoFar"],
                    #row["loss_pol_surr"],
                    row["loss_vf_loss"],
                    row["loss_kl"],
                    row["EpRewMean"],
                    row["EpLenMean"],
                    ])
                    ))
        w.write(html2)
    print("file://%s/%s.html" % (os.getcwd(), fn))
