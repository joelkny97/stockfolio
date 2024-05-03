document.addEventListener('DOMContentLoaded', function() {
    var rows = document.getElementById('portfolio-stocks');
    var data = [];
  
    // Iterate through each row in the table
    for (var row = 0; row < rows.getElementsByClassName('portfolio-stock-symbol').length; row++) {
      var label = rows.getElementsByClassName('portfolio-stock-symbol')[row].innerText.trim();
      var value = parseFloat(rows.getElementsByClassName('portfolio-stock-cost')[row].innerText.trim());
      data.push({ label: label, value: value });
    }
  
    // Create a color scale for the pie chart
    var color = d3.scaleOrdinal(d3.schemeCategory10);
  
    // Create a pie layout based on the data values
    var pie = d3.pie()
      .value(function(d) { return d.value; });
  
    // Create an arc generator for the pie slices
    var arc = d3.arc()
      .innerRadius(0)
      .outerRadius(150);
  
    // Select the SVG element where the pie chart will be drawn
    var svg = d3.select('#pie-chart')
      .append('svg')
      .attr('width', 300)
      .attr('height', 300)
      .append('g')
      .attr('transform', 'translate(150,150)'); // Center the pie chart
  
    // Generate pie chart data
    var arcs = svg.selectAll('arc')
      .data(pie(data))
      .enter()
      .append('g')
      .attr('class', 'arc');
  
    // Draw the pie slices
    arcs.append('path')
      .attr('d', arc)
      .attr('fill', function(d, i) { return color(i); })
      .attr('stroke', 'white')
      .attr('stroke-width', 2);
  
    // Add labels to the pie slices
    arcs.append('text')
      .attr('transform', function(d) {
        var centroid = arc.centroid(d);
        return 'translate(' + centroid[0] + ',' + centroid[1] + ')';
      })
      .attr('text-anchor', 'middle')
      .text(function(d) { return d.data.label; });
  });
  