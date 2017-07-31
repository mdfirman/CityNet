window.onload = function initMap() {

  function loadJSON(fname, callback) {

    var xobj = new XMLHttpRequest();
    xobj.overrideMimeType("application/json");
    xobj.open('GET', fname, true);
    xobj.onreadystatechange = function() {
      if (xobj.readyState == 4 && xobj.status == "200") {
        callback(xobj.responseText);
      }
    };
    xobj.send(null);
  }

  var london_centre = {lat: 51.529517, lng: -0.058284};

  //$("#myModel").load("./assets/sample_site.html");

  // We create the map in the json loader callback, beacuse javascript is strange
  loadJSON('style.json', function(response) {
    loaded_json = JSON.parse(response);
    var styledMapType = new google.maps.StyledMapType(loaded_json, {
      name: 'Map'
    });

    var map = new google.maps.Map(document.getElementById('map'), {
      zoom: 10,
      center: london_centre,
      mapTypeControlOptions: {
        mapTypeIds: ['satellite', 'styled_map']
      }
    });

    //Associate the styled map with the MapTypeId and set it to display.
    map.mapTypes.set('styled_map', styledMapType);
    map.setMapTypeId('styled_map');

    Papa.parse("assets/sites/sites_info.csv", {
    	download: true,
      dynamicTyping: true,
    	complete: function(results) {
        for (i=0; i<results.data.length - 1; i++)
        {
          icon_url = './assets/sites/charts/' + results.data[i][0] + '.png'

          var marker = new google.maps.Marker({
            position: {lat: results.data[i][1], lng: results.data[i][2]},
            map: map,
            icon: {
              url: './assets/charts/' + results.data[i][0] + '.png',
              scaledSize: new google.maps.Size(25, 25),
            },
            icon_path: './assets/charts/' + results.data[i][0] + '.png',
            postcode: ""+results.data[i][0],
          });

          console.log(results.data[i][0])

          google.maps.event.addListener(marker, 'mouseover', function() {
            this.setIcon({
              url: this.icon_path,
              scaledSize: new google.maps.Size(35, 35),
            });
          });

          google.maps.event.addListener(marker, 'mouseout', function() {
            this.setIcon({
              url: this.icon_path,
              scaledSize: new google.maps.Size(20, 20),
            });
          });

          marker.addListener('click', function() {
            console.log("Jere")
            $('#main-modal-title').html(this.postcode)
            $('#myModal').modal('show');
          });
        }
    	}
    });
    //
    // var map_inset = new google.maps.Map(document.getElementById('map_inset'), {
    //   center: london_centre,
    //   zoom: 18,
    //   mapTypeId: 'satellite',
    //   disableDefaultUI: true,
    //   scrollwheel: false,
    //   draggable: false
    // });
    //
    // // Resize map to show on a Bootstrap's modal
    // $('#myModal').on('shown.bs.modal', function() {
    //   var currentCenter = map_inset.getCenter();  // Get current center before resizing
    //   google.maps.event.trigger(map_inset, "resize");
    //   // console.log(london_centre)
    //   map_inset.setCenter(currentCenter); // Re-set previous center
    // });
    //

  });


}
