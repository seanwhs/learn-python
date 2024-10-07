function updateLabel(sliderId, labelId) {
    const slider = document.getElementById(sliderId);
    const label = document.getElementById(labelId);
    label.textContent = slider.value;
}

async function submitForm() {
    const form = document.getElementById('predictionForm');
    const formData = new FormData(form);
    const data = Object.fromEntries(formData.entries());

    // Convert neighbourhood_group and room_type to one-hot encoding
    const neighbourhoodGroup = ['Brooklyn', 'Manhattan', 'Queens', 'Staten Island'];
    const roomType = ['Private room', 'Shared room'];
    
    const oneHotNeighbourhoodGroup = neighbourhoodGroup.reduce((acc, group, index) => {
        acc[`neighbourhood_group_${group.replace(/ /g, '_')}`] = data['neighbourhood_group'] == index ? 1 : 0;
        return acc;
    }, {});

    const oneHotRoomType = roomType.reduce((acc, type, index) => {
        acc[`room_type_${type.replace(/ /g, '_')}`] = data['room_type'] == index ? 1 : 0;
        return acc;
    }, {});

    const payload = {
        minimum_nights: data['minimum_nights'],
        number_of_reviews: data['number_of_reviews'],
        reviews_per_month: data['reviews_per_month'],
        calculated_host_listings_count: data['calculated_host_listings_count'],
        availability_365: data['availability_365'],
        ...oneHotNeighbourhoodGroup,
        ...oneHotRoomType
    };

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });
        const result = await response.json();
        document.getElementById('result').innerHTML = `
            <h2>Prediction Result</h2>
            <p><strong>Prediction:</strong> ${result.prediction[0]}</p>
        `;
    } catch (error) {
        document.getElementById('result').innerHTML = `<p>Error: ${error.message}</p>`;
    }
}
