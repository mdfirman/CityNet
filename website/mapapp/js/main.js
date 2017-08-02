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

  var london_centre = {lat: 51.489517, lng: -0.124};

  // We create the map in the json loader callback, beacuse javascript is strange
  loadJSON('style.json', function(response) {
    loaded_json = JSON.parse(response);
    var styledMapType = new google.maps.StyledMapType(loaded_json, {
      name: 'Map'
    });

    var map = new google.maps.Map(document.getElementById('map'), {
      zoom: 11,
      center: london_centre,
      mapTypeControlOptions: {
        mapTypeIds: ['satellite', 'styled_map']
      }
    });

    //Associate the styled map with the MapTypeId and set it to display.
    map.mapTypes.set('styled_map', styledMapType);
    map.setMapTypeId('styled_map');

    var sz = 35;

    Papa.parse("assets/sites/sites_info.csv", {
    	download: true,
      dynamicTyping: true,
    	complete: function(results) {
        for (i=0; i<results.data.length - 1; i++)
        {
          var icon_url = './assets/sites/charts/' + results.data[i][0] + '.png';

          var marker = new google.maps.Marker({
            position: {lat: results.data[i][1], lng: results.data[i][2]},
            map: map,
            icon: {
              url: icon_url,
              scaledSize: new google.maps.Size(sz, sz),
            },
            icon_path: icon_url,
            postcode: results.data[i][0],
            wav_path: results.data[i][3],
            im_path: "assets/sites/ims/" + results.data[i][0] + ".jpg",
            chartdata_path: "assets/sites/chartdata/" + results.data[i][0] + ".json",
            description: results.data[i][4],
          });

          google.maps.event.addListener(marker, 'mouseover', function() {
            this.setIcon({
              url: this.icon_path,
              scaledSize: new google.maps.Size(sz+10, sz+10),
            });
          });

          google.maps.event.addListener(marker, 'mouseout', function() {
            this.setIcon({
              url: this.icon_path,
              scaledSize: new google.maps.Size(sz, sz),
            });
          });

          // When marer clicked, the modal is updated then shown
          marker.addListener('click', function() {

            $('#main-modal-title').html(this.description);
            // $('#main-modal-desc').html(this.postcode)

            var chart = Chartkick.charts["minute-data"];
            chart.updateData(this.chartdata_path);
            chart.redraw();
            // $.getJSON(this.chartdata_path, function(json) {
            //     console.log(this.chartdata_path);
            // });

            var myaudio = document.getElementById('audio_source');
            myaudio.src = 'assets/sites/audio/' + this.wav_path;

            var audio_container = document.getElementById('audio');
            audio_container.pause()
            audio_container.load()

            document.getElementById('site_image').src = this.im_path

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


  $('#myModal').on('hidden.bs.modal', function () {
    var audio_container = document.getElementById('audio');
    audio_container.pause()
  })


}
