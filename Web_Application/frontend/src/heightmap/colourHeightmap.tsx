import changeTexture2D from "../common/changeTexture2D";

export default function colourHeightmapHandler(params: any, mesh: THREE.Mesh<THREE.PlaneGeometry, THREE.MeshBasicMaterial>, texture: THREE.Texture, loader: THREE.TextureLoader, mapsArray: Array<Array<string>>, heightmapsArray: Float64Array[], mapsContainer: HTMLElement, heightmap2d: HTMLImageElement, texture2d: HTMLImageElement) {
    const colourHeightmap = () => {
        if (heightmap2d.src.slice(0, 22) !== 'data:image/png;base64,') {
            throw Error('No heightmap image src.')
        }
        fetch('http://127.0.0.1:8000/coloring', {
            method: 'POST',
            body: JSON.stringify({ 'image64': heightmap2d.src.slice(22), 'dataset': params.currentDataset, 'heightmap': heightmapsArray[params.currentId].toString() }),
            headers: {
                'Content-Type': 'application/json'
            },
        }).then(response => {
            console.log(response);
            response.json().then(res => {
                changeTexture2D(res.image64, params, mesh, texture, loader, mapsArray, mapsContainer, texture2d);
            })
        })
    }
    const buttonColourHeightmap = document.getElementById('btn_colour_heightmap') as HTMLElement;
    buttonColourHeightmap.addEventListener('click', colourHeightmap);
}