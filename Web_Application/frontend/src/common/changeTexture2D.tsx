export default function changeTexture2D(image64: string, params: any, mesh: THREE.Mesh<THREE.PlaneGeometry, THREE.MeshBasicMaterial>, texture: THREE.Texture, loader: THREE.TextureLoader, mapsArray: Array<Array<string>>, mapsContainer: HTMLElement, texture2d: HTMLImageElement) {
    const imgSrc = `data:image/png;base64,${image64}`;
    texture2d.src = imgSrc;
    texture = loader.load(imgSrc);
    mesh.material.map = texture;
    mapsArray[params.currentId][1] = imgSrc;
    const texturePreview = mapsContainer.childNodes[params.currentId].childNodes[1] as HTMLImageElement;
    texturePreview.src = imgSrc;
}