$(document).ready(function() {
    
    const ohlc = [],
     volume = [];
 

    var data = [];
    var rows = document.getElementById('portfolio-time-val-table');
    for (var row = 0; row < rows.getElementsByClassName('portfolio-table-value').length; row++) {
      var graph_point = [];
      var date = rows.getElementsByClassName('portfolio-table-date')[row].innerText.trim().split(/-/gm);
      graph_point.push(Date.UTC(parseFloat(date[0]), parseFloat(date[1]) - 1, parseFloat(date[2])));
      graph_point.push(parseFloat(rows.getElementsByClassName('portfolio-table-value')[row].innerText.trim().substr(1)));
      data.push(graph_point);
    }
    data.reverse();

    dataLength = data.length;

    console.log(data);

    for (let i = 0; i < dataLength; i += 1) {
        ohlc.push([
            data[i][1], // the date
            data[i][4], // open
            data[i][5], // high
            data[i][6], // low
            data[i][7] // close
        ]);

        volume.push([
            data[i][1], // the date
            data[i][3] // the volume
        ]);
    }
    Highcharts.stockChart('portfolio-ohlc', {
        yAxis: [{
            labels: {
                align: 'left'
            },
            height: '80%',
            resize: {
                enabled: true
            }
        }, {
            labels: {
                align: 'left'
            },
            top: '80%',
            height: '20%',
            offset: 0
        }],
        tooltip: {
            shape: 'square',
            headerShape: 'callout',
            borderWidth: 0,
            shadow: false,
            positioner: function (width, height, point) {
                const chart = this.chart;
                let position;

                if (point.isHeader) {
                    position = {
                        x: Math.max(
                            // Left side limit
                            chart.plotLeft,
                            Math.min(
                                point.plotX + chart.plotLeft - width / 2,
                                // Right side limit
                                chart.chartWidth - width - chart.marginRight
                            )
                        ),
                        y: point.plotY
                    };
                } else {
                    position = {
                        x: point.series.chart.plotLeft,
                        y: point.series.yAxis.top - chart.plotTop
                    };
                }

                return position;
            }
        },
        series: [{
            type: 'ohlc',
            id: 'aapl-ohlc',
            name: 'OHLC Plot',
            data: ohlc
        }, {
            type: 'column',
            id: 'aapl-volume',
            name: 'Volume',
            data: volume,
            yAxis: 1
        }],
        responsive: {
            rules: [{
                condition: {
                    maxWidth: 800
                },
                chartOptions: {
                    rangeSelector: {
                        inputEnabled: false
                    }
                }
            }]
        }
    });

    
  });
  